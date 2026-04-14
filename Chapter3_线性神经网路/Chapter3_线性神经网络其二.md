#### 3.4 softmax回归
- 分类问题。

##### 3.4.1 分类问题
- 热独编码（One-Hot Encoding）：将类别标签转换为向量的过程。每个类别对应一个向量，只有对应类别的位置为1，其余位置为0。

##### 3.4.2 网络架构
- 仿射函数（Affine Function）：线性函数加上偏置项的函数。形式为$\mathbf{w}^T\mathbf{x}+b$。

##### 3.4.3 全连接层的参数开销
- 将d个输入转换为q个输出的成本：O(dq/n)，其中n是批量大小。

##### 3.4.4 softmax运算
- 保证在任何数据上的输出都是非负的且综合为1以将输出视为概率。
- softmax函数：能偶将未规范化的预测变换为非负数且总和为1，同时让模型保持可导的性质，如下式：
$$\hat{\mathbf{y}}=\text{softmax}(\mathbf{o})，其中\hat{y}_j=\frac{e^{o_j}}{\sum_ke^{o_k}}$$
- $\forall j, 0 \leq \hat{y}_j \leq 1$，
- softmax回归是一个线性模型（Linear Model），因为softmax回归的输出仍然由输入特征的仿射变换决定。

##### 3.4.5 小批量样本的向量化
假设：读取了一个批量的样本$\mathbf{X}$，特征维度（输入数量）为d，批量大小为n，输出中有q个类别。对应削皮样本的特征为$\mathbf{X}\in\mathbb{R}^{n\times d}$，权重为$\mathbf{W}\in\mathbb{R}^{d\times q}$，偏置为$\mathbf{b}\in\mathbb{R}^q$，计算表达式：
$$\mathbf{O}=\mathbf{X}\mathbf{W}+\mathbf{b}$$
$$\hat{\mathbf{Y}}=\text{softmax}(\mathbf{O})$$

##### 3.4.6 损失函数
1. 对数似然
假设数据集$\{\mathbf{X},\mathbf{Y}\}$有n个样本，其中索引i的样本由特征向量$\mathbf{x}^{(i)}$和热独标签向量$\mathbf{y}^{(i)}$组成，将估计值与实际值进行比较：
$$P(\mathbf{Y}|\mathbf{X})=\prod_{i=1}^n P(\mathbf{y}^{(i)}|\mathbf{x}^{(i)})$$
最大化$P(\mathbf{Y}|\mathbf{X})$等价最小化负对数似然：
$$-\log P(\mathbf{Y}|\mathbf{X})=-\sum_{i=1}^nl(\mathbf{y}^{(i)},\hat{\mathbf{y}}^{(i)})$$
损失函数，即交叉熵损失（Cross-Entropy Loss）：
$$l(\mathbf{y},\hat{\mathbf{y}})=-\sum_{j=1}^q y_j \log \hat{y}_j$$
2. softmax及其导数
$$l(\mathbf{y},\hat{\mathbf{y}})=\log\sum_{k=1}^qe^{o_k}-\sum_{j=1}^qy_jo_j$$
得到：
$$\partial_{o_j}l(\mathbf{y},\hat{\mathbf{y}})=\frac{e^{o_j}}{\sum_{k=1}^qe^{o_k}}-y_j=\text{softmax}(\mathbf{o})_j-y_j$$

##### 3.4.7 信息论基础
1. 熵（Entropy）：量化数据中的信息内容内容，定义为：
$$H(P)=-\sum_jP(j)\log P(j)$$
2. 信息量：
$$\log\frac{1}{P(j)}=-\log P(j)$$
3. 交叉熵：“主观概率为Q的观察者在看到根据概率P生成的数据时的预期惊异”。

##### 3.4.8 模型预测和评估
- 精度（Accuracy）：预测书与预测总数的比例。

#### 3.5 图像分类数据集
数据集：Fashion-MNIST


```python
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from matplotlib_inline import backend_inline
from matplotlib import pyplot as plt

backend_inline.set_matplotlib_formats('svg')
```

##### 3.5.1 读取数据集
- 通过ToTensor实例将图像数据从PIL类型变换为32位浮点数并除以255使得所有图像的像素的数值均为0~1


```python
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False,transform=trans, download=True)
```

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to ../data\FashionMNIST\raw\train-images-idx3-ubyte.gz
    

    100%|██████████| 26.4M/26.4M [27:14<00:00, 16.2kB/s]
    

    Extracting ../data\FashionMNIST\raw\train-images-idx3-ubyte.gz to ../data\FashionMNIST\raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to ../data\FashionMNIST\raw\train-labels-idx1-ubyte.gz
    

    100%|██████████| 29.5k/29.5k [00:00<00:00, 36.6kB/s]
    

    Extracting ../data\FashionMNIST\raw\train-labels-idx1-ubyte.gz to ../data\FashionMNIST\raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to ../data\FashionMNIST\raw\t10k-images-idx3-ubyte.gz
    

    100%|██████████| 4.42M/4.42M [05:34<00:00, 13.2kB/s]
    

    Extracting ../data\FashionMNIST\raw\t10k-images-idx3-ubyte.gz to ../data\FashionMNIST\raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to ../data\FashionMNIST\raw\t10k-labels-idx1-ubyte.gz
    

    100%|██████████| 5.15k/5.15k [00:00<00:00, 34.4kB/s]
    

    Extracting ../data\FashionMNIST\raw\t10k-labels-idx1-ubyte.gz to ../data\FashionMNIST\raw
    
    

Fashion-MNIST数据集由10个类别的图像组成，每个类别由训练数据集（Train Dataset）中的6000张图像和测试数据集（Test Dataset）中的1000张图像组成。每个输入图像的高宽均为28像素，记为(28,28)。高度为h，宽度为w的像素图像形状记为(h,w)。


```python
len(mnist_train), len(mnist_test), mnist_train[0][0].shape
```




    (60000, 10000, torch.Size([1, 28, 28]))



10个类别分别为：t-shirt、trouser、pullover、dress、coat、sandal、shirt、sneaker、bag、ankle boot。


```python
def get_fashion_mnist_labels(labels): #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankel boot']

    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5): #@save
    """"绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图像张量
            ax.imshow(img.numpy())
        else:
            # PIL图像
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
```




    array([<Axes: title={'center': 'ankel boot'}>,
           <Axes: title={'center': 't-shirt'}>,
           <Axes: title={'center': 't-shirt'}>,
           <Axes: title={'center': 'dress'}>,
           <Axes: title={'center': 't-shirt'}>,
           <Axes: title={'center': 'pullover'}>,
           <Axes: title={'center': 'sneaker'}>,
           <Axes: title={'center': 'pullover'}>,
           <Axes: title={'center': 'sandal'}>,
           <Axes: title={'center': 'sandal'}>,
           <Axes: title={'center': 't-shirt'}>,
           <Axes: title={'center': 'ankel boot'}>,
           <Axes: title={'center': 'sandal'}>,
           <Axes: title={'center': 'sandal'}>,
           <Axes: title={'center': 'sneaker'}>,
           <Axes: title={'center': 'ankel boot'}>,
           <Axes: title={'center': 'trouser'}>,
           <Axes: title={'center': 't-shirt'}>], dtype=object)




    
![svg](Chapter3_%E7%BA%BF%E6%80%A7%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%85%B6%E4%BA%8C_files/Chapter3_%E7%BA%BF%E6%80%A7%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%85%B6%E4%BA%8C_8_1.svg)
    


##### 3.5.2 读取小批量


```python
batch_size = 256
def get_dataloader_workers(): #@save
    """使用4个进程来读取数据"""
    return 4
trin_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=get_dataloader_workers())
```

##### 3.5.3 整合所有组件


```python
def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,num_workers=get_dataloader_workers()), data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))
```

#### 3.6 softmax回归的从零开始实现


```python
from IPython import display
train_iter, test_iter = load_data_fashion_mnist(batch_size)
class Accumulator:
    """"在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

##### 3.6.1 初始化模型参数
10个类别，网络输出维度为10：权重构成784×10的矩阵，偏置构成1×10的行向量。使用正态分布初始化权重W，偏置初始化为0。


```python
num_inputs = 784
num_outputs = 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
```

##### 3.6.2 定义softmax操作
实现softmax的步骤：
- 对每个项求exp；
- 对每一行求和，得到每个样本的规范化常熟；
- 将每一行初一其规范化常熟，确保结果的和为1。
$$\text{softmax}(\mathbf{X})_{ij}=\frac{e^{\mathbf{X}_{ij}}}{\sum_k e^{\mathbf{X}_{ik}}}$$


```python
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition
```

##### 3.6.3 定义模型
定义输入通过网络映射到输出，将数据传递到模型之前使用reshape函数将每个原始图像展平为向量。


```python
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
```

##### 3.6.4 定义损失函数
依旧不使用for循环，而是通过一个运算符选择所有的元素。


```python
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

cross_entropy(y_hat, y)
```




    tensor([2.3026, 0.6931])



##### 3.6.5 分类精度
如果```y_hat```是矩阵，那么假定第二个维度存在存储每个类别的预测分数。使用```argmax```获得每行中最大元素的索引来预测类别，然后将预测类别与真实y元素比较。注意需要先将```y_hat```的数据类型转换为与y的数据类型一致。


```python
def accuracy(y_hat, y): #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter): #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for x, y in data_iter:
            metric.add(accuracy(net(x), y), y.numel())
    return metric[0] / metric[1]
```

##### 3.6.6 训练



```python
def train_epoch_ch3(net, train_iter, loss, updater): #@save
    """训练模型第一轮"""
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.SGD):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]
```

动画绘制


```python
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

class Animator: #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(3.5, 2.5)):
        # 增量绘制多条线
        if legend is None:
            legend = []
        backend_inline.set_matplotlib_formats('svg')
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
```

实现一个训练函数来训练模型，并在动画中绘制训练和测试的精度。


```python
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater): #@save
    """训练模型"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9], legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

lr = 0.1
def sgd(params, lr, batch_size):    #@save
    """小批量随机梯度下降"""
    with torch.no_grad():       # 禁用梯度跟踪，不构建计算图
        for param in params:
            param -= lr * param.grad / batch_size   # 更新参数
            param.grad.zero_()   # 清零梯度

def updater(batch_size):
    return sgd([W, b], lr, batch_size)
num_epochs = 50
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
```


    
![svg](Chapter3_%E7%BA%BF%E6%80%A7%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%85%B6%E4%BA%8C_files/Chapter3_%E7%BA%BF%E6%80%A7%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%85%B6%E4%BA%8C_30_0.svg)
    


##### 3.6.7 预测
给定一系列图像，比较它们的实际标签和模型预测的标签。


```python
def predict_ch3(net, test_iter, n=6):
    """预测标签"""
    for X, y in test_iter:
        break  # 只取一个 batch

    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(dim=1))

    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]

    show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
predict_ch3(net, test_iter)
```


    
![svg](Chapter3_%E7%BA%BF%E6%80%A7%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%85%B6%E4%BA%8C_files/Chapter3_%E7%BA%BF%E6%80%A7%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%85%B6%E4%BA%8C_32_0.svg)
    


#### 3.7 softmax回归的简洁实现


```python
from torch import nn
batch_size = 256
train_iter, test_iter= load_data_fashion_mnist(batch_size)
```

##### 3.7.1 初始化模型参数
PyTorch不会隐式地调整输入的形状，需要在线性层前定义展平层来调整网络输入的形状。


```python
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights);
```

##### 3.7.2 重新审视softmax的实现
为了避免$e^{o_k}$过大导致溢出，需要在softmax运算前执行$o_k-\max(o_k)$的操作。
$$\hat{y}_j=\frac{e^{o_j-\max(o_k)}e^{\max(o_k)}}{\sum_ke^{o_k-\max(o_k)}e^{\max(o_k)}}=\frac{e^{o_j-\max(o_k)}}{\sum_ke^{o_k-\max(o_k)}}$$
$$\log(\hat{y}_j)=o_j-\max(o_k)-\log\sum_ke^{o_k-\max(o_k)}$$


```python
loss = nn.CrossEntropyLoss(reduction='none')
```

##### 3.7.3 优化算法
使用学习率为0.1的小批量随机梯度下降作为优化算法。


```python
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```

##### 3.7.4 训练


```python
num_epochs = 50
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```


    
![svg](Chapter3_%E7%BA%BF%E6%80%A7%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%85%B6%E4%BA%8C_files/Chapter3_%E7%BA%BF%E6%80%A7%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%85%B6%E4%BA%8C_42_0.svg)
    

