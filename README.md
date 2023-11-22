# The source code of MTCN




# Train on Flickr30K
# t2i
python train.py --batch_size 64 --data_path data/ --dataset f30k --Matching_direction t2i --num_epochs 20

# i2t
python train.py --batch_size 128 --data_path data/ --dataset f30k --Matching_direction i2t --num_epochs 20
