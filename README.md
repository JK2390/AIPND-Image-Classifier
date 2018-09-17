# AIPND-Image-Classifier
This is an image classifier done with __Pytorch__, it's capable of using all types of *Resnet* as the pretrained model.

There is a script predict.py version which can be used to classify images, it has been created to classify *102 different species of flowers* but can be adapted easily to classify other types of images as well by pre-training the classifier and changing its output layer size for the desired number of categories.

The **flowers dataset** is available to *download* here:

https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz

## How to use the scripts

### TRAIN

```python train.py flowers --arch resnet18 --learning_rate 0.001 --epochs 5 --dropout 0.2 --gpu```


### PREDICT

```python predict.py /home/workspace/aipnd-project/flowers/test/18/image_04292.jpg checkpoints```
