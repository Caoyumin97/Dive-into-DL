# Task 01 线性回归-Softmax-MLP

## 1 线性回归

### 1.1 是什么

即单层神经网络，是最简单的机器学习模型之一：
$$
\bold y=x_1w_1+x_2w_2+...+x_nw_n+b=\bold x \bold w+b
$$
其中$\bold w$即权重，$b$即偏置。

### 1.2 模型训练

* 损失：均方误差

$$
L(\bold w,b)=\frac {1}{n} \sum_{i=1}^n\frac {1}{2}(\bold{w}^{T}\bold{x}^{(i)}+b-y^{(i)})^2
$$

* 优化：mini-batch stochastic gradient descent (SGD)
  $$
  (\bold{w},b) \leftarrow (\bold{w},b)-\frac{\alpha}{\left|{\beta}\right|} \sum_{i\in{\beta}}\partial_{(\bold w,b)}l^{(i)}(\bold w,b)
  \\
  or
  \\
  \bold\theta\leftarrow\theta-\frac{\alpha}{\left|{\beta}\right|}\nabla_{\bold\theta}l^{(i)}(\bold\theta)
  \\
  \nabla_{\bold\theta}l^{(i)}(\bold\theta)=\begin{bmatrix}
  x_1^{(i)} \\
  x_2^{(i)} \\
  1
  \end{bmatrix}(\hat{y}^{(i)}-y^{(i)})
  $$
  其中$\alpha$为学习率，$\beta$为批次大小

### 1.3 表示方法

from 《动手学深度学习》

![image-20200213210545649](C:\Users\DELL\Desktop\Dive-into-DL\Dive-into-DL\1.png)

### 1.0.0 学习代码

```python
# define a timer class to record time
class Timer(object):
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        # start the timer
        self.start_time = time.time()

    def stop(self):
        # stop the timer and record time into a list
        self.times.append(time.time() - self.start_time)
        return self.times[-1]

    def avg(self):
        # calculate the average and return
        return sum(self.times)/len(self.times)

    def sum(self):
        # return the sum of recorded time
        return sum(self.times)
```



## 2 Softmax

### 2.0 逻辑回归

* 信息熵

$$
H(X)=-\sum_{i=1}^{n}p(x_i)log(p(x_i))
$$

* 相对熵（Kullback-leibler散度）：描述两个概率分布的差距
  $$
  KL(P||Q)=\sum P(x)log\frac{P(x)}{Q(x)}
  $$

* 

$$
H(p,q)=-\sum{p(x_i)log(q(x_i))}
$$

<u>**相对熵=信息熵+交叉熵**</u>

* Logistic分布（$F(x)=\frac{1}{1+e^{-(x-\mu/\gamma)}}$）图形为“S形曲线”(sigmoid curve）

  ![image-20200213213908794](C:\Users\DELL\Desktop\Dive-into-DL\Dive-into-DL\2.png)

* 几率（odds）：某事件发生的概率与不发生的概率之比

* 对数几率（logit函数）：
  $$
  logit(p)=log\frac{p}{1-p}
  $$

* Logistic regression (for classification)
  $$
  logit(p)=\bold{wx}
  $$

### 2.1 是什么

单层神经网络的一种，线性回归用于连续值，softmax用于多输出离散值（分类任务）

### 2.2 模型运算

![image-20200213214539994](C:\Users\DELL\Desktop\Dive-into-DL\Dive-into-DL\3.png)
$$
\begin{aligned} o_1 &= x_1 w_{11} + x_2 w_{21} + x_3 w_{31} + x_4 w_{41} + b_1 \end{aligned}
\\
\begin{aligned} o_2 &= x_1 w_{12} + x_2 w_{22} + x_3 w_{32} + x_4 w_{42} + b_2 \end{aligned}
\\
\begin{aligned} o_3 &= x_1 w_{13} + x_2 w_{23} + x_3 w_{33} + x_4 w_{43} + b_3 \end{aligned}
\\
\hat{y}_1, \hat{y}_2, \hat{y}_3 = \text{softmax}(o_1, o_2, o_3)
\\
\hat{y}_1 = \frac{ \exp(o_1)}{\sum_{i=1}^3 \exp(o_i)},\quad \hat{y}_2 = \frac{ \exp(o_2)}{\sum_{i=1}^3 \exp(o_i)},\quad \hat{y}_3 = \frac{ \exp(o_3)}{\sum_{i=1}^3 \exp(o_i)}.
\\
or
\\
\begin{aligned} \boldsymbol{O} &= \boldsymbol{X} \boldsymbol{W} + \boldsymbol{b},\\ \boldsymbol{\hat{Y}} &= \text{softmax}(\boldsymbol{O}), \end{aligned}
$$

* 损失函数：交叉熵（cross entropy）
  $$
  H\left(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}\right ) = -\sum_{j=1}^q y_j^{(i)} \log \hat y_j^{(i)}
  \\
  \ell(\boldsymbol{\Theta}) = \frac{1}{n} \sum_{i=1}^n H\left(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}\right ),
  $$
  



## 3 MLP (Multi-Layer Perceptron)

### 3.1 是什么

多层感知机在单层神经网络的基础上引入了⼀到多个隐藏层（hidden layer）。隐藏层位于输入层和输出层之间。  

### 3.2 模型运算

$$
\begin{aligned} \boldsymbol{H} &= \boldsymbol{X} \boldsymbol{W}_h + \boldsymbol{b}_h,\\ \boldsymbol{O} &= \boldsymbol{H} \boldsymbol{W}_o + \boldsymbol{b}_o, \end{aligned}
$$

### 3.3 模型表示

![image-20200213215416433](C:\Users\DELL\Desktop\Dive-into-DL\Dive-into-DL\4.png)



