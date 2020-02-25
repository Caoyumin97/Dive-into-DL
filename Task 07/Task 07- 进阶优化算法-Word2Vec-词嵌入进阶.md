# Task 07- 进阶优化算法-Word2Vec-词嵌入进阶

## 1 进阶优化算法

### 1.1 Momentum

$$
\text{A example for steepest descent:}
\\
f(x_1,x_2)=0.1x_1^2+2x_2^2,\text{learning rate }{\alpha=0.4}
$$

![image-20200224114705380](C:\Users\DELL\Desktop\Dive-into-DL\Task 07\1.png)

> 同⼀位置上，目标函数在竖直方向（$x_2$轴方向）比在水平方向（$x_1$轴方向）的斜率的绝对值更大，因此，给定学习率，梯度下降迭代⾃变量时会使⾃变量在竖直⽅向⽐在⽔平⽅向移动幅度更⼤。那么，我们需要⼀个较小的学习率从而避免⾃变量在竖直⽅向上越过⽬标函数最优解。然而，这会造成⾃变量在⽔平⽅向上朝最优解移动变慢。  

* 在时间步$0$，动量法创建速度变量$v_0=0$；

* 在时间步$t>0$，动量法做如下迭代：
  $$
  v_t \leftarrow \gamma v_{t-1} + \eta_t g_t \ (0<\gamma<1) \Rightarrow \text{Exponentially Weighted Moving Average}
  \\
  x_t \leftarrow x_{t-1} - v_t
  \\
  \text{The same example with } {\gamma = 0.5}:
  $$

![image-20200224115358992](C:\Users\DELL\Desktop\Dive-into-DL\Task 07\2.png)

### 1.2 AdaGrad

> AdaGrad根据自变量在每个维度$(x_1,x_2,\cdots,x_n)$的梯度值的⼤小来调整各个维度上的学习率，从而避免统一的学习率难以适应所有维度的问题

$$
s_t \leftarrow s_{t-1} + g_t \odot g_t
\\
x_t \leftarrow x_{t-1} + \frac{\eta}{\sqrt{s_t+\epsilon}} \odot g_t
\\
{\odot}\text{: element-wise tensor multiplication}
$$

* 问题：由于$s_t$累加了元素平方的梯度，故学习率在一直降低/不变，在迭代后期可能会由于学习率国小而收敛过慢

### 1.3 RMSProp (AdaGrad + Exponentially WMA)

$$
s_t \leftarrow \gamma s_{t-1} + (1-\gamma)g_t \odot g_t
\\
x_t \leftarrow x_{t-1} + \frac{\eta}{\sqrt{s_t+\epsilon}} \odot g_t
$$

### 1.4 AdaDelta (RMSProp replace $\eta$ with $\sqrt{\Delta x_{t-1}}$)

$$
s_t \leftarrow \gamma s_{t-1} + (1-\gamma)g_t \odot g_t
\\
x_t \leftarrow x_{t-1} + \sqrt{\frac{\Delta x_{t-1}+\epsilon}{s_t+\epsilon}} \odot g_t
\\
\Delta x_{t} \leftarrow \rho\Delta x_{t-1}+(1-\rho)g_t^\prime\odot g_t
$$

### 1.5 Adam (RMSProp + Momentum)

$$
v_t \leftarrow \beta_1v_{t-1} + (1-\beta_1)g_t
\\
s_t \leftarrow \beta_2 s_{t-1} + (1-\beta_2)g_t \odot g_t
\\
\hat{v}_t \leftarrow \frac{v_t}{1-\beta_1^t},\hat{s}_t \leftarrow \frac{s_t}{1-\beta_2^t} \text{     （偏差修正）}
\\
x_t \leftarrow x_{t-1} + \frac{\eta\hat{v}_t}{\sqrt{\hat{s}_t}+\epsilon} \odot g_t
$$



## 2 word2vec

* 词向量：词的特征向量或表征，把词映射为实数域向量的技术叫词嵌入（word embedding）  

### 2.0 Why not One-Hot?

* One-Hot：每个向量长度等于词典大小$N$，词的索引为$i$，向量第$i$位为1

* 缺陷：词向量无法准确表达不同词之间的相似度（e.g., 任意两个词向量的余弦相似度为0）

### 2.1 word2vec

#### 2.1.1 跳字模型（skip-gram）

![image-20200224181042834](C:\Users\DELL\Desktop\Dive-into-DL\Task 07\3.png)

* 基于某个词（中心词）生成文本序列周围的词（背景词），参数为每个词所对应的中心词向量和背景词向量  
* 每个词表示为两个 $d$ 维向量，中心词向量 $v_i \in \mathbb{R}^d$，背景词向量 $u_i \in \mathbb{R}^d$
* 中心词 $w_c$，背景词 $w_o$

$$
P(w_o|w_c) = \frac{exp(u_o^{\top}v_c)}{\sum_{i\in \mathcal{V}}exp(u_i^{\top}v_c)}
$$

* 给定长度为$T$的文本序列，窗口大小为$m$，则给定任一中心词生成所有背景词的概率为：

$$
\prod_{t=1}^{T}\prod_{-m\leq{j}\leq{m},j\ne0}P(w^{(t+j)}|w^{(t)})
$$

* 训练：通过梯度下降训练需要计算$\mathcal{V}$中所有词作为$w_{c}$背景词的条件概率对参数进行更新

$$
\min_{\bold{v},\bold{u}}{L=-\prod_{t=1}^{T}\prod_{-m\leq{j}\leq{m},j\ne0}\log{P(w^{(t+j)}|w^{(t)})}}
\\
\log{P(w_o|w_c)} = u_o^{\top}v_c-\log{\sum_{i \in \mathcal{V}}\exp(u_i^{\top}v_c)}
\\
\frac{\partial \log P(w_o|w_c)}{\partial v_c} = u_o-\sum_{j \in \mathcal{V}}P(w_j|w_c)u_j
$$

* 使用skip-gram的中心词向量作为词的表征向量

#### 2.1.2 连续词袋模型（continuous bag of words, CBOW）

![image-20200224181059875](C:\Users\DELL\Desktop\Dive-into-DL\Task 07\4.png)

* 与skip-gram类似，区别在于CBOW基于背景词生成中心词

* 使用CBOW的背景词向量作为词的表征向量

#### 2.1.3 二次采样

* 文本数据中一般会出现一些高频词，如英文中的“the”“a”和“in”。通常来说，在一个背景窗口中，一个词（如“chip”）和较低频词（如“microprocessor”）同时出现比和较高频词（如“the”）同时出现对训练词嵌入模型更有益。因此，训练词嵌入模型时可以对词进行二次采样；
* 丢弃概率（越高频的词被丢弃的概率越大）：

$$
P(w_i)=\max(1-\sqrt{\frac{t}{f(w_i)}},0)
$$

* 代码

```python
def discard(idx):
    '''
    @params:
        idx: 单词的下标
    @return: True/False 表示是否丢弃该单词
    '''
    return random.uniform(0, 1) < 1 - math.sqrt(
        1e-4 / counter[idx_to_token[idx]] * num_tokens)
```

#### 2.1.4 负采样近似训练（negative sampling）

* 近似训练：输出概率之和本应为1，但计算开销太高（通过梯度下降训练需要计算$\mathcal{V}$中所有词作为$w_{c}$背景词的条件概率对参数进行更新），故通过采样的方法降低计算开销，但概率之和不严格等于1
* 负采样：对于一个中心词，背景词往往占总词典比例较小，而负样本（噪声词）的比例较大，故对负样本进行采样（噪声词采样概率$P(w)$设为词频与总词频之比的$0.75$次方）



## 3 词嵌入进阶

### 3.1 子词嵌入

> 英语单词通常有其内部结构和形成⽅式。例如，我们可以从“dog”“dogs”和“dogcatcher”的
> 字⾯上推测它们的关系。这些词都有同⼀个词根“dog”，但使⽤不同的后缀来改变词的含义。

* FastText：以固定大小的 n-gram 形式将单词更细致地表示为了子词的集合
* BPE (Byte Pair Encoding)：根据语料库的统计信息，自动且动态地生成高频子词的集合

### 3.2 GloVe全局向量词嵌入

1. 使用非概率分布的变量 $p_{ij}^{\prime}=x_{ij}$ 和 $q_{ij}^{\prime}=\exp⁡(u_j^{\top}v_i)$，并对它们取对数；

2. 为每个词 $w_i$ 增加两个标量模型参数：中心词偏差项 $b_i$ 和背景词偏差项 $c_i$，松弛了概率定义中的规范性；

3. 将每个损失项的权重 $x_i$ 替换成函数 $h(x_{ij})$，权重函数 $h(x)$ 是值域在 $[0,1]$ 上的单调递增函数，松弛了中心词重要性与 $x_i$ 线性相关的隐含假设；

4. 用平方损失函数替代了交叉熵损失函数：
   $$
   L=\sum_{i\in\mathcal{V}}\sum_{j\in\mathcal{V}} h(x_{ij}) (\boldsymbol{u}^\top_j\boldsymbol{v}_i+b_i+c_j-\log x_{ij})^2
   $$
   

