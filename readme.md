车牌号码识别，采用先分割后识别的方法进行

先使用分割网络对车牌进行分割，输入图像大小为 (512,512)，然后将分割得到的车牌输入到
识别网络中进行车牌号的识别,识别网络输入图像大小为(64,192)

使用的数据集车牌号的分布
![](https://github.com/youngx123/CarPlateRecognition/blob/master/img/data_distribution.png?raw=true)

识别中使用 Sigmoid 函数效果和使用Softmax函数效果对比