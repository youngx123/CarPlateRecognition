车牌号码识别，采用先分割后识别的方法进行

先使用分割网络对车牌进行分割，输入图像大小为 (512,512)，然后将分割得到的车牌输入到
识别网络中进行车牌号的识别,识别网络输入图像大小为(64,192)

使用的数据集车牌号的分布
![](https://github.com/youngx123/CarPlateRecognition/blob/master/img/data_distribution.png?raw=true)

在车牌号码识别中, 若 `ctc=True`,则使用`torch.nn.CTCLoss()`损失函数，可以忽略由于蓝牌和新能源车牌造成的车牌号位数不一致的问题。
不需要在数据生成器中额外对车牌号进行对齐处理。
```python
def RecgTrain_fit(imagesize:tuple, bachsize=None, epoch=None, device=None, modelName=None)
    ctc = True # # False
    model = Recg(image_size=imagesize, ctc=ctc)
    ...
```
测试时对网络结果进行解码后处理
```python
def CTC_Decode(pred):
    _, pred = pred.max(2)
    pred = pred.transpose(1,0).contiguous().view(-1)
    char_list = []
    for i in range(len(pred)):
        if pred[i]!=0 and (not (i>0 and pred[i-1]==pred[i])):
            char_list.append(pred[i].item())
    
    char_list1 = [i-1 for i in char_list]
    char_list2 = [c - provincesNum for c in char_list[1:]]
    result = [char_list1[0]] + char_list2
    return "_".join([str(i) for i in result])
```


`loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)`

log_probs : 模型输出结果，经过 `log_softmax` 函数处理， `(T, B, C)`, `T` 为序列的长度,`B` 为批处理大小,`C` 预测类别数.

targets : 所有标签合并后的结果， 维度为 `(s)`

input_lengths : 由 序列长度组成的 `B`个元素， [T]*B

target_lengths : 每个标签的长度

识别中使用 Sigmoid 函数效果和使用Softmax函数效果对比

车牌识别结果：(数据集问题导致，省份识别结果较差)
```python
input label :  0_0_0_0_24_30_26 蓝牌
pred  label :  0_0_0_0_24_30_26 蓝牌
****************************************
input label :  0_0_0_0_24_31_30 蓝牌
pred  label :  0_0_0_0_24_31_30 蓝牌
****************************************
input label :  0_0_0_0_25_32_27 蓝牌
pred  label :  0_0_0_0_25_32_27 蓝牌
****************************************
input label :  0_0_0_0_26_24_24 蓝牌
pred  label :  0_0_0_0_26_24_24 蓝牌
****************************************
input label :  0_0_0_0_27_28_27 蓝牌
pred  label :  0_0_0_0_27_28_27 蓝牌
****************************************
input label :  0_0_0_1_26_28_31 蓝牌
pred  label :  0_0_0_1_26_28_31 蓝牌
****************************************
input label :  26_6_33_0_33_24_33 蓝牌
pred  label :  0_6_33_0_33_24_33 蓝牌
****************************************
input label :  26_7_13_13_33_31_31 蓝牌
pred  label :  0_7_13_13_33_31_31 蓝牌
****************************************
input label :  27_0_23_15_33_31_30 蓝牌
pred  label :  0_0_23_15_33_31_30 蓝牌
****************************************
input label :  27_11_1_7_29_29_32 蓝牌
pred  label :  0_11_1_7_29_29_32 蓝牌
****************************************
input label :  28_0_10_2_29_30_32 蓝牌
pred  label :  0_0_10_2_29_30_32 蓝牌
****************************************
input label :  29_0_5_25_31_29_29_29 绿牌
pred  label :  0_0_5_25_31_29_29_29 绿牌
****************************************
```
