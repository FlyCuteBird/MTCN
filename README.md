# Introduction
Source code of the MTCN.
## Requirements
The following dependencies are recommended.

* Python==3.7.0
* pytorch==1.7.0
* torchvision==0.8.0
* torchaudio==0.7.0
* pytorch-pretrained-bert==0.6.2
  
## Pretrained model
If you don't want to train from scratch, you can download the pre-trained model from [here](https://drive.google.com/drive/folders/1LizeREOYUHdpoDzFAMyqzGDPi_s0WaQW?usp=drive_link) (for Flickr30K)
```bash
i2t: 487.4
Image to text: 73.2  92.5  96.4
Text to image: 55.6  81.8  87.9
t2i: 497.9
Image to text: 75.3  94.3 97.4
Text to image: 57.7  83.6  89.6
```
## Download Data 
We utilize the image feature created by SCAN, downloaded [here](https://github.com/kuanghuei/SCAN). Some related text data can be found [here](https://drive.google.com/drive/folders/1y55ccAlmoT7VSnNzLBYLPI-oYNRKX--K?usp=drive_link).

```
# Train on Flickr30K
python train.py --batch_size 64 --data_path data/ --dataset f30k --Matching_direction t2i --num_epochs 20
python train.py --batch_size 128 --data_path data/ --dataset f30k --Matching_direction i2t --num_epochs 20

## Evaluation
Run ```test.py``` to evaluate the trained models on f30k.
```
