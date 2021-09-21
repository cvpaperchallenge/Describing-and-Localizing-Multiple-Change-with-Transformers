# Multiple Change Captioning Code

- This directory includes code for generating multiple change captioning.

For training MCCFormers-D:

```
python train_trans.py --data_folder $DATA_FOLDER --encoder 'MCCFormers-D' --decoder 'trans' --feature_dim_de 1024
```

For training MCCFormers-S:
```
python train_trans.py --data_folder $DATA_FOLDER --encoder 'MCCFormers-S' --decoder 'trans' --feature_dim_de 1024
```
