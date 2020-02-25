# Task 05- CNN-LeNet-CNN进阶

## 1 卷积神经网络（CNN, Convolutional Neural Network）

### 1.1 基础

* 含有卷积层的神经网络
* 卷积层并非采用严格的数学卷积，而是采用了**互相关**（cross-correlation）运算，即<u>矩阵按元素求积求和+滑动窗</u>

### 1.2 特征图与感受野

* 特征图：⼆维卷积层输出的⼆维数组可以看作是输⼊在空间维度（宽和⾼）上某⼀级的表征
* 感受野：影响元素$x$的前向计算的所有可能输⼊区域（可能大于输入的实际尺寸）



## 2 LeNet-AlexNet-VGG-NiN-GoogleNet

* LeNet（1994）

![img](C:\Users\DELL\Desktop\Dive-into-DL\Task 05\1.jpg)

* AlexNet（2012）

![img](C:\Users\DELL\Desktop\Dive-into-DL\Task 05\2.jpg)

* VGG（2014）

![img](C:\Users\DELL\Desktop\Dive-into-DL\Task 05\3.png)

* NiN，Network in Network（2014）
* GoolgeNet（2014）：Inception模块

![image-20200216131740139](C:\Users\DELL\Desktop\Dive-into-DL\Task 05\4.png)