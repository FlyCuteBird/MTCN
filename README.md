# The source code of MTCN

```
# Train on Flickr30K
python train.py --batch_size 64 --data_path data/ --dataset f30k --Matching_direction t2i --num_epochs 20
python train.py --batch_size 128 --data_path data/ --dataset f30k --Matching_direction i2t --num_epochs 20

## Evaluation
Run ```test.py``` to evaluate the trained models on f30k.
```
