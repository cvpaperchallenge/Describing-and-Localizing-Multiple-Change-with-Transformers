import json
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models_trans import MHAFF, VisualTransformer, DecoderTransformer, PlainDecoder, M_VAM, DualAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu ##--

from torch.autograd import Variable

import argparse

# Data parameters
data_name = '3dcc_5_cap_per_img_0_min_word_freq'

# Model parameters 
embed_dim = 512
decoder_dim = 512
dropout = 0.5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
captions_per_image = 5


# Training parameters
start_epoch = 0
batch_size = 64
workers = 1
decoder_lr = 1e-4
encoder_lr = 1e-4
grap_clip = 5.
best_bleu4 = 0.
print_freq = 100

para_lambda1 = 1.0
para_lambda2 = 1.0

def get_key(dict_, value):
  return [k for k, v in dict_.items() if v == value]

def main(args):

  torch.manual_seed(0)
  torch.cuda.manual_seed_all(0)
  torch.backends.cudnn.benckmark = False
  torch.backends.cudnn.deterministic = True

  global start_epoch, data_name

  # Read word map
  word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + data_name + '.json')
  with open(word_map_file, 'r') as f:
    word_map = json.load(f)

  # Initialize
  if args.encoder == 'MHAFF':
    encoder = MHAFF(feature_dim = 1024,dropout=0.5,h=16,w=16,d_model=512,n_head=args.n_head,n_layers=args.n_layers).to(device)

  if args.encoder == 'VIT':
    encoder = VisualTransformer(feature_dim = 1024,h=16,w=16, n_head=args.n_head,n_layers=args.n_layers).to(device)

  if args.encoder == 'DUDA':
    encoder = DualAttention(attention_dim=args.attention_dim, feature_dim = args.feature_dim).to(device)

  if args.encoder == 'VAM':
    encoder = M_VAM().to(device)

  if args.decoder == 'trans':
    decoder = DecoderTransformer(feature_dim = args.feature_dim_de,
                               vocab_size = len(word_map),
                               n_head = args.n_head,
                               n_layers = args.n_layers,
                               dropout=dropout).to(device)


  if args.decoder == 'plain':
    decoder = PlainDecoder(feature_dim = args.feature_dim_de,
                           embed_dim = embed_dim,
                           vocab_size = len(word_map),
                           hidden_dim = args.hidden_dim,
                           dropout=dropout).to(device)


  
  encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                       lr=encoder_lr)

  decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                       lr=decoder_lr)

  criterion = nn.CrossEntropyLoss().to(device)

  train_loader = torch.utils.data.DataLoader(
    CaptionDataset(args.data_folder, data_name, 'TRAIN', captions_per_image, args.dataset_name),
    batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

  for epoch in range(start_epoch, args.epochs):
    print("epoch : " + str(epoch))

    train(train_loader=train_loader,
          encoder=encoder,
          decoder=decoder,
          criterion=criterion,
          encoder_optimizer=encoder_optimizer,
          decoder_optimizer=decoder_optimizer,
          epoch=epoch,
          word_map=word_map
          )

    # Save checkpoint
    save_checkpoint(args.root_dir, data_name, epoch, encoder, decoder, encoder_optimizer, decoder_optimizer)
    

def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, word_map):
  encoder.train()
  decoder.train()

  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  top3accs = AverageMeter()

  start = time.time()

  # Batches
  for i, (imgs1, imgs2, caps, caplens) in enumerate(train_loader):
    data_time.update(time.time() - start)

    # Move to GPU, if available
    imgs1 = imgs1.to(device)
    imgs2 = imgs2.to(device)
    caps = caps.to(device)
    caplens = caplens.to(device)

    # Forward prop.
    l = encoder(imgs1, imgs2)
    scores, caps_sorted, decode_lengths, sort_ind = decoder(l, caps, caplens)
    
    targets = caps_sorted[:, 1:]

    scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
    targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

    loss = criterion(scores, targets)

    # Back prop.
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss.backward()

    # Update weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    # Keep track of metrics
    top3 = accuracy(scores, targets, 3)
    losses.update(loss.item(), sum(decode_lengths))
    top3accs.update(top3, sum(decode_lengths))
    batch_time.update(time.time() - start)

    start = time.time()
    
    # Print status
    if i % print_freq == 0:
      print('Epoch: [{0}][{1}/{2}]\t'
            'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Top-3 Accuracy {top3.val:.3f} ({top3.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                    batch_time=batch_time,
                                                                    data_time=data_time,
                                                                    loss=losses,
                                                                    top3=top3accs))

if __name__=='__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--data_folder', default='dataset/for_train/3dcc_v0-2/concat_subtract')
  parser.add_argument('--root_dir', default='results/rcc_v0-2_total_subtract')
  parser.add_argument('--hidden_dim', type=int, default=512)
  parser.add_argument('--attention_dim', type=int, default=512)
  parser.add_argument('--epochs', type=int, default=41)
  parser.add_argument('--encoder', default='MHAFF')
  parser.add_argument('--decoder', default='trans')
  parser.add_argument('--n_head', type=int, default=4)
  parser.add_argument('--n_layers', type=int, default=2)
  parser.add_argument('--feature_dim', type=int, default=1024)
  parser.add_argument('--feature_dim_de', type=int, default=1024)
  parser.add_argument('--dataset_name', default='MOSCC')

  args = parser.parse_args()

  main(args)






