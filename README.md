# key points estimation and point instance segmentation approach for lane detection
link:[https://arxiv.org/abs/2002.06604](https://arxiv.org/abs/2002.06604)  

## Acknowledgement
The project refers to the following projects:
* [PINet_new](https://github.com/koyeongmin/PINet_new#key-points-estimation-and-point-instance-segmentation-approach-for-lane-detection) 

## Dependency
python3 ( python 3.7.8 )

pytorch ( pytorch 1.6.0 with GPU(RTX2080ti) )

opencv

numpy

sklearn

ujon

csaps

tqdm


## My work
* Implement quantize aware training in pytorch

* Crop out the sky area and reconstruct the lane dataset

* Redesign the backbone with just 238M FLOPs(original PINet 4G FLOPs) and the result is not decrease


## Datasets
We train and test the model on our own dataset in the same format as the Tusimple dataset.

## Test
We provide trained model, and it is saved in "savefile" directory. You can test as following,

`sh pinet_quantization_test/test.sh`

It has some mode like following functions.
mode 1 : Run the model on the given video. If you want to use this mode, enter your video path at line 63 in "test.py"

mode 2 : Run the model on the given image. If you want to use this mode, enter your image path at line 82 in "test.py"

mode 3 : Test the model on whole test set, and save result as json file.

You can change mode at line 22 in "parameters.py".

## Train
You can train the model as following:

`sh pinet_quantization_train/train.sh`

## 
The testing results are as follows:

![pic](https://github.com/Derteanoo/PINet_quantization/blob/master/pinet_quantization_test/test_pic_res/city_day3.mp4_20200818_165445248.jpg）


