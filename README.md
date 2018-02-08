# style_transfer_tensorflow
A tensorflow implementation for [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155).
In this project, I build a model with Tensorflow and slim for image style transfer.

## Result:
| sample | origin |
| :---: | :---: |
| ![](https://github.com/benbenlijie/style_transfer_tensorflow/blob/master/img/styled_test.jpg)|  ![](https://github.com/benbenlijie/style_transfer_tensorflow/blob/master/img/test.jpg)  |
| ![](https://github.com/benbenlijie/style_transfer_tensorflow/blob/master/img/styled_test1.jpg)|  ![](https://github.com/benbenlijie/style_transfer_tensorflow/blob/master/img/test1.jpg)  |
| ![](https://github.com/benbenlijie/style_transfer_tensorflow/blob/master/img/styled_test2.jpg)|  ![](https://github.com/benbenlijie/style_transfer_tensorflow/blob/master/img/test2.jpg)  |
| ![](https://github.com/benbenlijie/style_transfer_tensorflow/blob/master/img/styled_test3.jpg)|  ![](https://github.com/benbenlijie/style_transfer_tensorflow/blob/master/img/test3.jpg)  |



## Train a Model:

### Preparetion:
1. download [VGG16 model](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz) from Tensorflow Slim. Extract the file vgg_16.ckpt. Then copy it to the folder pretrained/
2. download [COCO dataset](http://msvocds.blob.core.windows.net/coco2014/train2014.zip). Please unzip it.
3. convert the COCO images to tfrecord file by follow command:
```
python data_loader/dataset_create.py
```  
It will create a ```image.tfrecord``` file placed at folder ```datasets```

### Train:
```
python transfer.py -c conf/candy.yml
``` 

### Tensorboard:
```
tensorboard --logdir logs/candy
```

### Model checkpoints
saved at ```models/candy```

## Tansfer a picture with trained model:

```
python evaluate.py -c conf/candy.yml -i img/test.jpg
```
A style transfered image named ```styled_test.jpg``` will be created and placed at the same folder as the input image.