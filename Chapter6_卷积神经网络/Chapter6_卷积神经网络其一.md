### 6 卷积神经网络
卷积神经网络（Convolutional Neural Network，CNN），是一类强大的、为处理图像数据而设计的神经网络。
#### 6.1 从全连接层到卷积
##### 6.1.1 不变性
卷积神经网络将 **空间不变性（Spatial Invariance）** 的这一概念系统化，从而基于这个模型使用较少的参数来学习有用的表示。
- **平移不变性（Translation Invariance）**：不管检测对象出现在哪个位置，神经网路的前面几层应该对应相同的图像区域具有相似的反应，即平移不变性。
- **局部性（Locality）**：神经网络的前面几层应该只探索输入图像中的局部区域，而不过度在意图像中相隔较远区域的关系，这就是“局部性”原则。最终可以聚合这些局部特征，以在整张图像级进行预测。

##### 6.1.2 多层感知机的限制
使用$[\mathbf{X}]_{i,j}$和$[\mathbf{H}]_{i,j}$分别表示输入图像和隐藏表示中位置$(i,j)$处的像素。为了使每个隐藏神经元都能接收每个输入像素的信息，将参数从权重矩阵替换为四阶权重张量W。假设$\mathbf{U}$包含偏执参数，可以将全连接层形式化表示为
$$[\mathbf{H}]_{i,j}=[\mathbf{U}]_{i,j}+\sum_k\sum_l[W]_{i,j,k,l}[\mathbf{X}]_{k,l}=[\mathbf{U}]_{i,j}+\sum_a\sum_b[V]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$$
用相对位置偏移$(a,b)$替代绝对位置$(k,l)$，表示输入像素时围绕$(i,j)$的一个偏移。
- **全连接层完全忽略图像的空间结构**。
1. 平移不变性
检测对象在输入$\mathbf{X}$中的平移，应该仅导致隐藏表示$\mathbf{H}$中的平移，即$V$和$\mathbf{U}$不依赖于$(i,j)$的值，$[V]_{i,j,a,b}=[\mathbf{V}]_{a,b}$，可以简化$\mathbf{H}$定义：
$$[\mathbf{H}]_{i,j}=u+\sum_a\sum_b[\mathbf{V}]_{a,b}[\mathbf{X}]_{i+a,j+b}$$
这就是**卷积（Convolution）**，使用系数$[\mathbf{V}]_{a,b}$对位置$(i,j)$附近的像素$(i+a,j+b)$进行加权得到$[\mathbf{H}]_{i,j}$，且不依赖于图像中的位置。
2. 局部性
为了收集用来训练参数$[\mathbf{H}]_{i,j}$的相关信息，不应偏离到距$(i,j)$很远的位置，即在$|a|>\Delta$或$|b|>\Delta$的范围，可以设置$[\mathbf{V}]_{a,b}=0$，此时：
$$[\mathbf{H}]_{i,j}=u+\sum_{a=-\Delta}^{\Delta}\sum_{b=-\Delta}^{\Delta}[\mathbf{V}]_{a,b}[\mathbf{X}]_{i+a,j+b}$$
上式为一个卷积层（Convolutional Layer），卷积神经网络是包含卷积层的一类特殊的神经网络。$[\mathbf{V}]$成为**卷积核（Convolution Kernel）** 或**滤波器（Filter）**，或者简单地称为该卷积层的**权重**，通常该权重是科学系的参数。当图像处理的局部区域很小时，卷积神经网络与多层感知机的训练擦会议可能是巨大的：多层感知机可能需要数十亿个参数来表示网络中的一层，而卷积神经网路通常只需要数百个参数，而且通常不需要改变输入或隐藏表示的维数。

##### 6.1.3 卷积
卷积的定义$(f,g:\mathbb{R}^d\rightarrow\mathbb{R})$：
$$(f*g)(\mathbf{x})=\int f(\mathbf{\tau})g(\mathbf{x}-\mathbf{\tau})d\mathbf{\tau}$$
即卷积是当把一个函数“反转”并移位$\mathbf{x}$时，测量$f$和$g$之间的重叠。对于二维张量可以如下表示：
$$(f*g)(i,j)=\sum_a\sum_b f(a,b)g(i-a,j-b)$$

##### 6.1.4 “沃尔多在哪里”
目标：学习一个模型，检测出“沃尔多”最可能出现的地方。
- **通道**
对于彩色图像，包含3个通道，是一个由高度、宽度、颜色组成的三维张量，如下表示：
$$[\mathbf{H}]_{i,j,d}=u+\sum_{a=-\Delta}^{\Delta}\sum_{b=-\Delta}^{\Delta}\sum_c[\mathbf{V}]_{a,b,c,d}[\mathbf{X}]_{i+a,j+b,c}$$

#### 6.2 图像卷积
##### 6.2.1 互相关运算
卷积层实际表达的运算是**互相关（Cross-Correlation）**，在卷积层中，输入张量和核张量通过互相关运算生成输出张量。在图像中移动卷积核，保持输出大小不变，使用```corr2d```函数实现该功能，接收输入张量X和卷积核张量K，返回输出张量Y。举例：
$$\left[
\begin{array}{ccc}
    0 & 1 & 2 \\
    3 & 4 & 5 \\
    6 & 7 & 8 \\
\end{array}\right]*\left[\begin{array}{cc}
    0 & 1 \\
    2 & 3 \\
\end{array}\right]=\left[\begin{array}{cc}
    19 & 25 \\
    37 & 43 \\
\end{array}\right]$$
输出大小略小于输入大小，输出大小等于输入大小$n_h\times n_w$减去卷积核大小$k_h\times k_w$，即
$$(n_h-k_h+1)\times(n_w-k_w+1)$$


```python
import torch
from torch import nn
def corr2d(X, K):   #@save
    """二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y
X = torch.tensor([[0,1,2], [3,4,5], [6,7,8]])
K = torch.tensor([[0,1], [2,3]])
corr2d(X, K)
```




    tensor([[19., 25.],
            [37., 43.]])



##### 6.2.2 卷积层
卷积层对输入和卷积核进行互相关运算，添加标量偏置后产生输出。卷积层中的两个被训练的参数是卷积核和标量偏置。


```python
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

##### 6.2.3 图像中的目标边缘检测
通过找到像素变化的位置来检测图像中不同颜色的边缘。


```python
X = torch.ones((6, 8))
X[:, 2:6] = 0
X
```




    tensor([[1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.]])




```python
K = torch.tensor([[1.0, -1.0]]) # 卷积核
K
```




    tensor([[ 1., -1.]])




```python
Y = corr2d(X, K)
Y
```




    tensor([[ 0.,  1.,  0.,  0.,  0., -1.,  0.],
            [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
            [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
            [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
            [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
            [ 0.,  1.,  0.,  0.,  0., -1.,  0.]])




```python
corr2d(X.t(), K)
```




    tensor([[0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]])



##### 6.2.4 学习卷积核
对于更复杂的情况，不可能手动设计卷积核。可以通过仅查看“输入-输出”对来学习由X生成Y的卷积核。

先构造一个卷积层，并将其卷积核初始化为随机张量，在每次迭代中，比较Y与卷积层输出的平方误差，然后计算梯度来更新卷积核。

- 前向传播：
$$\hat{\mathbf{Y}}_{i,j}=\sum_{a,b}\mathbf{V}_{a,b}\mathbf{X}_{i+a,j+b}$$
- 损失函数：
$$L=\sum_{i,j}(\hat{\mathbf{Y}}_{i,j}-\mathbf{Y}_{i,j})^2$$
- 反向传播：
$$\frac{\partial L}{\partial \mathbf{V}_{a,b}}=\sum_{i,j}2(\hat{\mathbf{Y}}_{i,j}-\mathbf{Y}_{i,j})\cdot\mathbf{X}_{i+a,j+b}$$
- 更新参数：
$$\mathbf{V}_{a,b}\leftarrow\mathbf{V}_{a,b}-\eta\frac{\partial L}{\partial \mathbf{V}_{a,b}}$$


```python
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)
# 四维输入输出：批量大小、通道、高度、宽度
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2
for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum()}:.3f')
```

    epoch 2, loss 15.531808853149414:.3f
    epoch 4, loss 4.972557067871094:.3f
    epoch 6, loss 1.803678274154663:.3f
    epoch 8, loss 0.6996816396713257:.3f
    epoch 10, loss 0.2800287902355194:.3f
    


```python
conv2d.weight.data.reshape((1, 2))
```




    tensor([[ 0.9333, -1.0417]])



##### 6.2.5 互相关和卷积
卷积核是从数据中学习的，执行卷积运算和互相关运算，卷积层输出不受影响。

##### 6.2.6 特征映射和感受野
- 特征映射（Feature Map）：可以被视为一个输入映射到下一层的空间维度的转换器。
- 感受野（Receptive Field）：对于某一层的输入x，在前向传播期间可能影响x计算的所有元素。

#### 6.3 填充和步幅
一个240×240像素的图像经过10层5×5的卷积后，减少到200×200像素，会丢失信息，引入填充（Padding）。又或者，原始的输入分辨率冗余，希望减少宽高，引入步幅（Stride）。
##### 6.3.1 填充
填充：在图像的边缘填充元素，通常是0：
$$(n_h-k_h+1)\times(n_w-k_w+1)\Rightarrow(n_h-k_h+p_h+1)\times(n_w-k_w+p_w+1)$$
且通常设置$p_h=k_h-1，p_w=k_w-1$。

定义一个函数，初始化卷积层权重，并对输入和输出扩大和缩减相应的维数。


```python
def comp_conv2d(conv2d, X):
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])   # 省略批量大小和通道两个维度
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
comp_conv2d(conv2d, X).shape
```




    torch.Size([8, 8])



##### 6.3.2 步幅
为了高效计算或是缩减采样次数，卷积窗口可以跳过中间位置，每次滑动多个元素。当垂直和水平步幅为$(s_h,s_w)$时，输出形状为：
$$\lfloor(n_h-k_h+p_h+s_h)/s_h\rfloor\times\lfloor(n_w-k_w+p_w+s_w)/s_w\rfloor$$
当$p_h=k_h-1，p_w=k_w-1$，可简化为：
$$\lfloor(n_h+s_h-1)/s_h\rfloor\times\lfloor(n_w+s_w-1)/s_w\rfloor$$


```python
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape
```




    torch.Size([2, 2])



#### 6.4 多输入多输出通道
每个RGB输入图像具有3×h×w的形状，将这个大小为3的轴称为**通道（Channel）** 维度。
##### 6.4.1 多输入通道
对每个通道执行互关操作，然后将结果相加。


```python
def corr2d_multi_in(X, K):
    return sum(corr2d(x, k) for x, k in zip(X, K))
X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])
corr2d_multi_in(X, K)
```




    tensor([[ 56.,  72.],
            [104., 120.]])



##### 6.4.2 多输出通道
可以将每个通道看作对不同特征的响应。在互相关运算中，每个通道先获取所有输入通道，再以对应该输出通道的卷积核计算出结果。


```python
def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)
K = torch.stack((K, K + 1, K + 2), 0)
K.shape
```




    torch.Size([3, 2, 2, 2])




```python
corr2d_multi_in_out(X, K)
```




    tensor([[[ 56.,  72.],
             [104., 120.]],
    
            [[ 76., 100.],
             [148., 172.]],
    
            [[ 96., 128.],
             [192., 224.]]])



##### 6.4.3 1×1卷积层
当的输入和输出具有相同的高度和宽度，输出中的每个元素都是输入图像中同一位置的元素的线性组合。可以将1×1卷积层看作在每个像素位置应用的全连接层，以$c_i$个输入值转换为$c_o$个输出值。1×1卷积层需要的权重维度为$c_o\times c_i$，再额外加上一个偏置。


```python
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))
X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6
```

#### 6.5 汇聚层
希望在处理图像时逐渐降低隐藏表示的空间分辨率、聚合信息，这样随着神经网路中层数的增加，每个神经元对其敏感的感受野（输入）就越大。通过逐渐聚合信息，生成越来越粗粒度的映射，最终实现学习全局表示的目标，同时将卷积层的所有优势保留在中间。

当检测较低层的特征时，通常希望这些特征保持某种程度的平移不变性。为此，引入**汇聚层（Pooling Layer）**：降低卷积层对位置的敏感性，同时降低对空间降采样表示的敏感性。
##### 6.5.1 最大汇聚和平均汇聚
汇聚层运算符由一个固定形状的窗口组成，该窗口根据其步幅大小在输入的所有区域上滑动，为固定形状的窗口遍历的每个位置计算一个输出。汇聚层不包含参数，汇聚操作是确定性的，通常计算汇聚窗口中所有元素的最大值或平均值，分别成为**最大汇聚（Maximum Pooling）** 和**平均汇聚（Average Pooling）**。


```python
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i + p_h, j:j + p_w].max()
            elif mode == "avg":
                Y[i, j] = X[i:i + p_h, j:j + p_w].mean()
    return Y
X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]).float()
pool2d(X, (2, 2)), pool2d(X, (2, 2), 'avg')
```




    (tensor([[4., 5.],
             [7., 8.]]),
     tensor([[2., 3.],
             [5., 6.]]))



##### 6.5.2 填充和步幅


```python
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4, ))
X
```




    tensor([[[[ 0.,  1.,  2.,  3.],
              [ 4.,  5.,  6.,  7.],
              [ 8.,  9., 10., 11.],
              [12., 13., 14., 15.]]]])




```python
pool2d = nn.MaxPool2d(3)
pool2d(X)
```




    tensor([[[[10.]]]])




```python
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```




    tensor([[[[ 5.,  7.],
              [13., 15.]]]])




```python
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
pool2d(X)
```




    tensor([[[[ 5.,  7.],
              [13., 15.]]]])



##### 6.5.3 多个通道
在处理多通道输入数据时，汇聚层在每个输入通道上单独运算，而不是像卷积层那样在通道上对输入进行汇总。即汇聚层的输出通道数与输入通道数相同。


```python
X = torch.cat((X, X + 1), 1)
X
```




    tensor([[[[ 0.,  1.,  2.,  3.],
              [ 4.,  5.,  6.,  7.],
              [ 8.,  9., 10., 11.],
              [12., 13., 14., 15.]],
    
             [[ 1.,  2.,  3.,  4.],
              [ 5.,  6.,  7.,  8.],
              [ 9., 10., 11., 12.],
              [13., 14., 15., 16.]]]])




```python
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```




    tensor([[[[ 5.,  7.],
              [13., 15.]],
    
             [[ 6.,  8.],
              [14., 16.]]]])


