 ### 5 深度学习计算
 #### 5.1 层和块
- 单个神经元完成的工作：接收输入；生成相应的标量输出；具有一组相关参数（Parameter），更新参数优化目标函数。
- 层完成的工作：接收一组输入；生成相应的输出；由一组可调整参数描述。
- 神经网路块（Block）：描述单个层、由多个层组成的组件或整个模型本身

##### 5.1.1 自定义块
基本功能：
- 将输入数据作为其前向传播函数的参数；
- 通过前向传播函数来生成输出；
- 计算器输出关于输入的梯度，可通过其反向传播函数仅从访问；
- 存储和访问前向传播计算所需的参数；
- 根据需要初始化参数模型


```python
import torch
from torch import nn
from torch.nn import functional as F
class MLP(nn.Module):   # 定义一个神经网络模型类，继承自nn.Module
    # 声明两个全连接层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化
        # 在实例化时也可以指定其他函数参数，如模型参数params
        super().__init__()                  # 调用nn.Module的初始化
        self.hidden = nn.Linear(20, 256)    # 隐藏层
        self.out = nn.Linear(256, 20)       # 输出层
    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 使用nn.functional模块定义的ReLU函数
        return self.out(F.relu(self.hidden(X)))
```

前向传播函数以X作为输入→进入隐藏层```self.hidden(X)```→激活```F.relu(...)```→输出```self.out(...)```，等价于
$$\mathbf{Y}=(\max(0, \mathbf{XW}_1+b_1))\mathbf{W}_2+b_2$$

##### 5.1.2 顺序块
定义两个函数：
- 将块逐个追加到列表中的函数；
- 前向传播函数，用于将输入按追加块的顺序传递给组成的“链条”。


```python
class MySequential(nn.Module):  # 定义一个容器类，类似nn.Sequential(l1, l2)
    def __init__(self, *args):  # *args表示可以传入任意多个模块
        super().__init__()
        # 注册模块，把每一层存进self._modules，key=0,1,2....，value是具体的层
        # _modules是用于存子模块的字典
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module
    def forward(self, X):       # 前向传播
        for block in self._modules.values():    # 顺序连接多个网络，链式传播
            X = block(X)
        return X
```

##### 5.1.3 在前线传播函数中执行代码
##### 5.1.4 效率

#### 5.2 参数管理
- 访问参数，用于调试、诊断和可视化；
- 参数初始化；
- 在不同模型组件间共享参数。


```python
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)
```




    tensor([[0.2083],
            [0.1287]], grad_fn=<AddmmBackward0>)



##### 5.2.1 参数访问


```python
print(net[2].state_dict())  # 检查第二个全连接层的参数
```

    OrderedDict([('weight', tensor([[-0.2310,  0.3028, -0.3479,  0.1607, -0.0073,  0.0797,  0.1676, -0.2784]])), ('bias', tensor([0.1818]))])
    

1. 目标参数


```python
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
```

    <class 'torch.nn.parameter.Parameter'>
    Parameter containing:
    tensor([0.1818], requires_grad=True)
    tensor([0.1818])
    

2. 一次性访问所有参数


```python
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
```

    ('weight', torch.Size([8, 4])) ('bias', torch.Size([8]))
    ('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8])) ('2.weight', torch.Size([1, 8])) ('2.bias', torch.Size([1]))
    

3. 从嵌套快收集函数
将多个块嵌套。


```python
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())
def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}', block1())
    return net
rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet(X))
print(rgnet)
```

    tensor([[-0.3161],
            [-0.3171]], grad_fn=<AddmmBackward0>)
    Sequential(
      (0): Sequential(
        (block 0): Sequential(
          (0): Linear(in_features=4, out_features=8, bias=True)
          (1): ReLU()
          (2): Linear(in_features=8, out_features=4, bias=True)
          (3): ReLU()
        )
        (block 1): Sequential(
          (0): Linear(in_features=4, out_features=8, bias=True)
          (1): ReLU()
          (2): Linear(in_features=8, out_features=4, bias=True)
          (3): ReLU()
        )
        (block 2): Sequential(
          (0): Linear(in_features=4, out_features=8, bias=True)
          (1): ReLU()
          (2): Linear(in_features=8, out_features=4, bias=True)
          (3): ReLU()
        )
        (block 3): Sequential(
          (0): Linear(in_features=4, out_features=8, bias=True)
          (1): ReLU()
          (2): Linear(in_features=8, out_features=4, bias=True)
          (3): ReLU()
        )
      )
      (1): Linear(in_features=4, out_features=1, bias=True)
    )
    

对于分层嵌套的层，可以像通过嵌套列表索引一样访问层。


```python
rgnet[0][1][0].bias.data
```




    tensor([-0.2814, -0.4081,  0.2553, -0.2747,  0.4869, -0.0867,  0.3254,  0.0736])



##### 5.2.2 参数初始化
1. 内置初始化


```python
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
print(net[0].weight.data[0], net[0].bias.data[0])
```

    tensor([0.0137, 0.0047, 0.0021, 0.0064]) tensor(0.)
    


```python
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
print(net[0].weight.data[0], net[0].bias.data[0])
```

    tensor([1., 1., 1., 1.]) tensor(0.)
    


```python
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)
net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0], net[2].weight.data)
```

    tensor([-0.4967, -0.4281,  0.0735, -0.5304]) tensor([[42., 42., 42., 42., 42., 42., 42., 42.]])
    

2. 自定义初始化
使用以下分布为任意权重参数w定义初始化方法：
$$w\sim\left\{\begin{array}{ll}
    U(5,19),    可能性为\frac{1}{4}\\
    0,          可能性为\frac{1}{2}\\
    U(-10,-5),  可能性为\frac{1}{4}\end{array}\right.$$


```python
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5
net.apply(my_init)
net[0].weight[:2]
```

    Init weight torch.Size([8, 4])
    Init weight torch.Size([1, 8])
    




    tensor([[-0.0000, -0.0000,  0.0000, -5.6351],
            [ 7.6080,  8.0630, -0.0000,  0.0000]], grad_fn=<SliceBackward0>)



直接设置参数：


```python
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```




    tensor([42.0000,  2.0000,  2.0000, -3.6351])



##### 5.2.3 参数绑定
希望在多个层间共享参数，可以定义一个稠密层，然后使用这个稠密层的参数来设置另一个层的参数。


```python
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8),
                    nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
print(net[2].weight.data[0] == net[4].weight.data[0])
```

    tensor([True, True, True, True, True, True, True, True])
    tensor([True, True, True, True, True, True, True, True])
    

#### 5.3 延后初始化
目前没有做到的事情：
- 定义了网络架构，但是没有指定输入维度；
- 添加层时没有指定前一层的输出维度；
- 初始化参数时，没有足够的信息来确定模型应该包含多少参数

解决方法：延后初始化（Defer Initialization），即直到数据第一次通过模型传递时，框架才会动态地推断出每个层的大小。

#### 5.4 自定义层
用创造性的方式组合不同的层，从而设计出适用于各种任务的架构。
##### 5.4.1 不带参数的层
构造一个没有任何参数的层。下面构建的```CenteredLayer```类要从其输入中减去均值，只需继承基本层类并实现前向传播功能。


```python
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X):
        return X - X.mean()
layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))
```




    tensor([-2., -1.,  0.,  1.,  2.])




```python
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))
Y.mean()
```




    tensor(9.3132e-10, grad_fn=<MeanBackward0>)



##### 5.4.2 带参数的层
使用内置函数来创建参数，这些函数提供了管理访问、初始化、共享、保存和加载模型参数等功能。使用修正线性单元作为激活函数，需要输入参数```in_units```和```units```，分别表示输入数和输出数。


```python
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
linear = MyLinear(5, 3)
linear.weight
```




    Parameter containing:
    tensor([[ 0.5218,  0.0577, -0.8162],
            [ 0.0309, -0.3360,  0.6079],
            [ 1.5052, -0.7568,  1.2951],
            [-0.3323, -0.4075,  0.5642],
            [-0.9351, -0.4410, -1.6076]], requires_grad=True)



#### 5.5 读写文件
保存训练的模型，以备将来在各种环境中使用。同时，在一个耗时较长的训练过程时，可以定期保存中间结果，避免损失。
##### 5.5.1 加载和保存张量
对于单个张量，可以直接调用```load```和```save```函数分别读写，这两个函数都要求提供一个名称，```save```要求将要保存的变量作为输入。


```python
x = torch.arange(4)
torch.save(x, 'x-file')
x2 = torch.load('x-file', weights_only=True)
print(x2)
```

    tensor([0, 1, 2, 3])
    

注意：使用```torch.load(..., weights_onlyj=True)```，有执行外来恶意代码的风险。


```python
y = torch.zeros(4)
torch.save([x, y], 'x-files')
x2, y2 = torch.load('x-files', weights_only=True)
(x2, y2)
```




    (tensor([0, 1, 2, 3]), tensor([0., 0., 0., 0.]))




```python
mydict = {'x': x, 'y':y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict', weights_only=True)
mydict2
```




    {'x': tensor([0, 1, 2, 3]), 'y': tensor([0., 0., 0., 0.])}



##### 5.5.2 加载和保存模型参数
深度学习框架提供了内置函数来保存和加载整个网络，保存模型的参数而不是整个模型。


```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)
    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))
net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
torch.save(net.state_dict(), 'mlp.params')
```

为了恢复模型，实例化原始多层感知机模型的一个备份，不需要随机初始化模型参数，直接读取文件中存储的参数。


```python
clone = MLP()
clone.load_state_dict(torch.load('mlp.params', weights_only=True))
clone.eval()
```




    MLP(
      (hidden): Linear(in_features=20, out_features=256, bias=True)
      (output): Linear(in_features=256, out_features=10, bias=True)
    )



#### 5.6 GPU
