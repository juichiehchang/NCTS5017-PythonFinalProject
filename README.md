# NCTS5017-PythonFinalProject

## Description

This is the python final project for **NCTS5017 Machine-learning** class.  
This repository contains the *contruction* and *training* codes of the two models used for **NSFW image classification**.  
All the	images are downloaded with scripts from	[nsfw_data_scraper](https://github.com/alexanderkim-io/nsfw_data_scraper).

There are a total of **210190** images after running `cleanse.py` to remove all the corrupted images (truncated, exif data error...).  

These are the numbers of images in each class : 

- `drawings` : Normal drawings including anime - **33508**
- `hentai` : NSFW drawings including hentai - **42698**
- `neutral` : Normal pictures including buildings and people - **48661**
- `porn` : NSFW pornography pictures - **69183**
- `sexy` : Sexually explicit pictures, but no nudity - **16140**

Two different models are used to build the neural networks: `InceptionV3` and `DenseNet121`, codes can be found in the separate directories with their name.

**88%** accuracy was achieved using the `InceptionV3` model.

## Implementation

- InceptionV3  
  The **InceptionV3  classfier** is built upon the `keras` `InceptionV3` model with weights pretrained on `ImageNet`. Several `pooling` and `dense` layers are added to improve its accuracy. All the input images are resized to  `(299x299x3)` , which is the default size for the `InceptionV3` input model.
  
- DenseNet121  
  The **DenseNet121 classifier** is built upon the `keras` `DenseNet121` model with weights pretrained on `ImageNet`. Several `pooling` and `dense` layers are added to improve its accuracy. All the input images are resized to `(224x224x3)` , which is the default size for the `DenseNet121` input model.
  
For the training of both models. `Batch_size` is set to *32* and there are *500* batches per `epoch`, which means *16000* images will be processed in each `epoch`. The `InceptionV3` model was able to achieve **88%** accuracy in about 25 epochs.

## Usage

Thanks to 陳怡升 and 李志恆, we were able to contruct a **NSFW-detector** program. The program will automatically scan your screen regularly. If any NSFW image is detected, it will mess with your mouse, play the [screaming cowboy part](https://www.youtube.com/watch?v=Qcp2W1-SFt4) in *Big Enough* and automatically close the tab after 15 seconds. You can check out the example video on youtube [here](https://www.youtube.com/watch?v=AsDYYk-qPA8&feature=youtu.be).

**juichieh.chang 張睿傑**