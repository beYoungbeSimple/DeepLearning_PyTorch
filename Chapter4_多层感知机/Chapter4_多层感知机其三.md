#### 4.7 前向传播、反向传播和计算图
前期的学习中，只考虑了通过前向传播（Forward Propagation）所设计的计算。再计算梯度时，至调用了深度学习框架提供的反向传播函数，而不知其所以然。
##### 4.7.1 前向传播
前向传播：按顺序（从输入层到输出层）计算和存储神经网路中每层的结果。假设输入样本时$\mathbf{x}\in\mathbb{R}^d$，隐藏层不包括偏置项，则此处中间变量为：
$$\mathbf{z}=\mathbf{W}^{(1)}\mathbf{x}$$

其中$\mathbf{W}^{(1)}\in\mathbb{R}^{h\times d}$是隐藏层的权重参数。将中间变量$\mathbf{z}\in\mathbb{R}^h$通过激活函数$\phi$后得到长度为$\text{h}$的隐藏层激活向量：
$$\mathbf{h}=\phi(\mathbf{x})$$

隐藏层激活向量$\mathbf{h}$也是中间变量。假设输出层的参数只有权重$\mathbf{W}^{(2)}\in\mathbb{R}^{q\times h}$，可以得到长度为$\text{q}$的输出层变量：
$$\mathbf{o}=\mathbf{W}^{(2)}\mathbf{h}$$

损失函数$\text{l}$，样本标签$\text{y}$，计算单个数据样本的损失项：
$$L=l(\mathbf{o},y)$$

根据$\text{L}_2$正则化，给定超参数$\lambda$，正则化项为：
$$s=\frac{\lambda}{2}\left(\|\mathbf{W}^{(1)}\|_F^2+\|\mathbf{W}^{(2)}\|_F^2\right)$$

最后得到正则化损失为：
$$J=L+s=l(\mathbf{o},y)+\frac{\lambda}{2}\left(\|\mathbf{W}^{(1)}\|_F^2+\|\mathbf{W}^{(2)}\|_F^2\right)$$

##### 4.7.2 前向传播计算图

##### 4.7.3 反向传播
反向传播（Backward Propagation/Backpropagation）：一种计算神经网络参数梯度的方法，根据微积分的链式法则，按相反的顺序从输出层到输入层遍历网络，存储了计算某些参数梯度时所需的任何中间变量（偏导数）。假设有函数$\mathbf{Y}=f(\mathbf{X})$和$\mathbf{Z}=g(\mathbf{Y})$，应用链式法则计算$\mathbf{Z}$的偏导数：
$$\frac{\partial \mathbf{Z}}{\partial \mathbf{X}}=\text{prod}(\frac{\partial \mathbf{Z}}{\partial \mathbf{Y}},\frac{\partial \mathbf{Y}}{\partial \mathbf{X}})$$

对于正则化损失
$$J=L+s=l(\mathbf{o},y)+\frac{\lambda}{2}\left(\|\mathbf{W}^{(1)}\|_F^2+\|\mathbf{W}^{(2)}\|_F^2\right)$$
有：
$$\frac{\partial J}{\partial L}=1，\frac{\partial J}{\partial S}=1$$
$$\frac{\partial J}{\partial\mathbf{o}}=\frac{\partial L}{\partial\mathbf{o}}\in\mathbb{R}^q$$
$$\frac{\partial J}{\partial\mathbf{W}^{(1)}}=\frac{\partial J}{\partial\mathbf{z}}\mathbf{x}^T+\lambda\mathbf{W}^{(1)}$$
$$\frac{\partial J}{\partial\mathbf{W}^{(2)}}=\frac{\partial J}{\partial\mathbf{o}}\mathbf{h}^T+\lambda\mathbf{W}^{(2)}$$
$$\frac{\partial s}{\partial\mathbf{W}^{(1)}}=\lambda\mathbf{W}^{(1)}，\frac{\partial s}{\partial\mathbf{W}^{(2)}}=\lambda\mathbf{W}^{(2)}$$

##### 4.7.4 训练神经网络
对于前向传播，沿着依赖的方向遍历计算图并计算其路径上的所有变量，然后用于反向传播。在训练神经网络时，初始化模型参数后，交替使用前向传播和反向传播，利用反向传播给出的梯度来更新模型参数。注意反向传播重复利用前向传播中存储的中间值避免重复计算——需要更多的内存。

##### **维度规则**
- 前向传播维度规则
$$\mathbf{Y}=\mathbf{XW}+b$$
$$\left[
\begin{array}{ccc}
    y_{11}  & \cdots & y_{1,d_{out}} \\
    \vdots  & \ddots & \vdots       \\
    y_{batch,1}  & \cdots & y_{batch,d_{out}} \\
\end{array}\right]=\left[
\begin{array}{ccc}
    x_{11}  & \cdots & x_{1,d_{in}} \\
    \vdots  & \ddots & \vdots       \\
    x_{batch,1}  & \cdots & x_{batch,d_{in}} \\
\end{array}\right]\left[
\begin{array}{cccc}
    w_{11} & \cdots & w_{1,d_{out}} \\
    \vdots & \ddots & \vdots \\
    w_{d_{in},1} & \cdots & w_{d_{in},d_{out}} \\
\end{array}\right]+\left[
\begin{array}{ccc}
    b_1\\
    \vdots\\
    b_{out}\end{array}\right]
$$
- 反向传播维度规则
$$\frac{\partial L}{\partial\mathbf{Y}}=\mathbf{G}_{(batch\times d_{out})}$$
$$\frac{\partial L}{\partial\mathbf{X}}=\mathbf{G}_{(batch\times d_{out})}\mathbf{W}_{(d_{in}\times d_{out})}^T$$
$$\frac{\partial L}{\partial\mathbf{W}}=\mathbf{X}_{(batch\times d_{in})}^T\mathbf{G}_{(batch\times d_{out})}$$
$$\frac{\partial L}{\partial b}=\sum_{batch}\mathbf{G}_{(batch\times d_{out})}$$

#### 4.8 数值稳定性和模型初始化
梯度可视作$L-1$个矩阵$\mathbf{M}^{(L)}\cdots\mathbf{M}^{(l+1)}$与梯度向量$\mathbf{v}^{(l)}$的乘积。
1. 梯度消失
梯度消失（Gradient Vanishing）问题：参数更新过小，在每次更新时几乎不会移动，导致模型无法学习。对于sigmoid函数，当输入过小或过大时，梯度会消失。
2. 梯度爆炸
梯度爆炸问题（Gradient Exploding）问题：参数更新过大，破坏了模型的稳定性收敛。若初始化不当，可能导致没有机会让梯度下降优化器收敛。
3. 打破对称性
在每一层的隐藏单元之间具有排列对称性，小批量随机梯度下降不能打破这种对称性，但暂退法和正则化可以。

##### 4.8.1 参数初始化
1. 默认初始化
不指定初始化方法，框架将使用默认的随机初始化方法，对于中等难度的问题，这种方法通常有效。
2. Xavier初始化
对于没有非线性的全连接层输出$o_i$的分布：
$$o_i=\sum_{j=1}^{n_{in}}=w_{ij}x_j$$
计算均值和方差：
$$\text{E}[o_i]=\sum_{j=1}^{n_{in}}E[w_{ij}]E[x_j]，\text{Var}[o_i]=\sum_{j=1}^{n_{in}}E[w_{ij}^2]E[x_j^2]=n_{in}\sigma^2\gamma^2$$

需要满足：
$$\frac{1}{2}(n_{in}+n_{out})\sigma^2=1$$
得到Xavier初始化分布：
$$U(-\sqrt{\frac{6}{n_{in}+n_{out}}},\sqrt{\frac{6}{n_{in}+n_{out}}})$$

#### 4.9 环境和分布偏移
数据最初从哪来？最终如何处理模型的输出？
##### 4.9.1 分布偏移的类型
1. 协变量偏移
协变量偏移（Covariate Shift）假设虽然输入的分布可能随时间改变，但标签函数（即条件分布$P(y|\mathbf{x})$没有改变
2. 标签偏移
标签偏移（Label Shift）：与协变量偏移相反，假设标签边缘概率$P(y)$可以改变，但是类别条件分布$P(y|\mathbf{x})$在不同的领域保持不变。
3. 概念偏移
概念偏移（Concept Shift）：类别会随着不同时间的用法而改变。


