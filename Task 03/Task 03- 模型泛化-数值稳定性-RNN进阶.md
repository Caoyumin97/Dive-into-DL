# Task 03- 模型泛化-数值稳定性-RNN进阶

## 1 模型泛化

* **数据集划分**：训练集（/验证集）/测试集

* **泛化性能**（狭义）：通过训练集训练得到的模型，在测试集上的性能

  * 训练误差：模型在训练集上表现出的误差
  * 泛化误差：模型在任意⼀个测试数据样本上表现出的误差期望，常通过测试集上的误差来近似

* **模型选择**：从训练集中预留验证集进行参数优化与模型选择

  * $K$-折交叉验证（$K$-$fold$）

  ![img](C:\Users\DELL\Desktop\Dive-into-DL\1.png)

  * 留一法交叉验证（$Leave$-$One$-$Out$）：样本容量为$N$，每次留出$1$个样本作为验证集，剩余$(N-1)$个样本作为训练集，则共开发$N$个模型，最终泛化误差为各个样本泛化误差均值

* 模型复杂度&过拟合-欠拟合

  ![image-20200215122553852](C:\Users\DELL\Desktop\Dive-into-DL\2.png)
  * 欠拟合改善方法：特征工程、增加模型复杂度、增强拟合能力

  * 过拟合改善方法：

    * 增大数据集：收集数据、数据增广

    * 权重衰减（$weight$ $decay$）：$L2$正则化，惩罚绝对值较⼤的模型参数
      $$
      l = l(w_1,w_2,b)+\frac{\lambda}{2n}{\left\| w \right\|}^2
      $$

    * $Dropout$：在一个训练轮次中，随机将一个神经网络层中的若干个神经元无效化（不更新参数），从而使得模型不依赖于特定的神经元

* 计算图

  ![image-20200215123627785](C:\Users\DELL\Desktop\Dive-into-DL\3.png)



## 2 数值稳定性

### 2.1 梯度稳定性

根据反向传播机制，梯度根据链式法则由后向前递乘，容易出现梯度过小（gradient vanishing）与梯度过大的现象（gradient exploding）

### 2.2 参数初始化

* Xavier随机初始化：全连接层输入维度为$a$，输出维度为$b$，则每个权重元素的初始值随机采样于均匀分布$U(-\sqrt{\frac{6}{a+b}},\sqrt{\frac{6}{a+b}}).$
* He初始化：（适用于ReLU激活函数），每个权重元素采样于正态分布$N(0,\sqrt\frac{2}{a})$

### 2.3 偏移

* 协变量偏移（$covariate$ $shift$）：训练/测试集的输入特征分布差异较大

* 标签偏移（$label$ $shift$）：训练/测试集的标签分布差异较大

* 概念偏移：标签的定义发生变化？



## 3 RNN进阶

### 3.1 通过时间反向传播（$BPTT,$ $Back$ $Propagation$ $Through$ $Time$）

* 计算图（无偏置项+线性激活）

  ![image-20200215130510827](C:\Users\DELL\Desktop\Dive-into-DL\4.png)

* 运算

  * 模型定义（定义前向传播）

  $$
  \bold{h}_{t} = \bold W_{xh}\bold{x}_{t} + \bold W_{hh}\bold{h}_{t-1}
  \\
  \bold{o}_t=\bold{W}_{qh}\bold{h}_t
  \\
  L=\frac{1}{T}\sum_{t=1}^{T}l(\bold{o}_t,\bold{y}_t)
  $$

  * 反向传播
    $$
    \frac{\partial L}{\partial W_{qh}}=\frac{1}{T}\sum_{t=1}^{T} [\frac{\partial l(\bold {o}_t,\bold{y}_t)}{\partial \bold {o}_t} \frac{\partial\bold{o}_t}{\partial W_{qh}}]=\frac{1}{T}\sum_{t=1}^{T}\frac{\partial l(\bold {o}_t,\bold{y}_t)}{\partial \bold {o}_t}\bold{h}_t^{\top}
    \\
    \frac{\partial L}{\partial W_{xh}} = \frac{\partial L}{\partial \bold{o}_t} \frac{\partial \bold{o}_t}{\partial \bold{h}_t} \frac{\partial \bold{h}_t}{\partial W_{xh}} = \frac{1}{T}\sum_{t=1}^{T} \frac{\partial l(\bold {o}_t,\bold{y}_t)}{\partial \bold {o}_t} W_{qh}x_t^{\top}
    \\
    \frac{\partial L}{\partial W_{hh}} = \frac{\partial L}{\partial \bold{o}_t} \frac{\partial \bold{o}_t}{\partial \bold{h}_t} \frac{\partial \bold{h}_t}{\partial W_{hh}} = \frac{1}{T}\sum_{t=1}^{T} \frac{\partial l(\bold {o}_t,\bold{y}_t)}{\partial \bold {o}_t} W_{hh}h_{t-1}^{\top}
    $$

  * 机理

    与MLP的计算图不同，MLP的反向传播可以高度并行化，如均方误差损失+线性激活的Perceptron，其$\frac{\partial L}{\partial w_{a,b}}=|o_b-y_b|x_a$只依赖于输入$x_a$而不依赖于其他状态，而RNN的计算图由于存在序列上的依赖关系，所以反向传播时，需要依赖于若干时间步长之前的状态来对梯度进行计算（见$\frac{\partial L}{\partial W_{hh}}$）

### 3.2 GRU

* 门控循环单元，Gated Recurrent Unit
* 门控：重置门（reset gate）和更新门（update gate）

* 计算图

![image-20200215154848990](C:\Users\DELL\Desktop\Dive-into-DL\5.png)

### 3.3 LSTM

* 长短期记忆单元，Long Short Term Memory
* 门控：输入门（input gate），遗忘门（forget gate），输出门（output gate）
* 计算图

![image-20200215155038506](C:\Users\DELL\Desktop\Dive-into-DL\6.png)

### 3.4 deep & bi-directional

* 深层RNN

  ![image-20200215155214877](C:\Users\DELL\Desktop\Dive-into-DL\7.png)

* 双向RNN：隐藏状态的传递方向包含了前向后及后向前

  ![image-20200215155234553](C:\Users\DELL\Desktop\Dive-into-DL\8.png)