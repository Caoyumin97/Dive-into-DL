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

![image-20200213210545649](C:\Users\DELL\AppData\Roaming\Typora\typora-user-images\image-20200213210545649.png)

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

