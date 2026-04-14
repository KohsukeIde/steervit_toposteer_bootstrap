Please download the data and organize them in the folders are indicated below.
##### Imagenet100 and Imagnet1k
The structure should be as follows:
```
dataset root.
│   imagenet100.txt
└───train
│   └─── n*
│       │   *.tar
│       │   ...
│   └─── n*
│    ...
```
The path to the dataset root is to be used in the configs.

##### COCO
The structure for training should be as follows:
```
dataset root.
└───images
│   └─── train2017
│       │   *.jpg
│       │   ...
```

##### VOC Pascal
The structure for training and evaluation should be as follows:
```
dataset root.
└───SegmentationClass
│   │   *.png
│   │   ...
└───SegmentationClassAug # contains segmentation masks from trainaug extension 
│   │   *.png
│   │   ...
└───images
│   │   *.jpg
│   │   ...
└───sets
│   │   train.txt
│   │   trainaug.txt
│   │   val.txt
```