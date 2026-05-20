#### 7.3 网络中的网络（NiN）
LeNet、AlexNet、VGG都是通过一些列的卷积层与汇聚层来提取空间机构特征，然后通过全连接层对特征的表征进行处理。使用了全连接层，可能会放弃表征的空间结构。网络中的网络（Net in Net，NiN）提供了一个简单的解决方案：在每个像素的通道上分别使用多层感知机。
##### 7.3.1 NiN块
四维张量：样本、通道、高度、宽度；全连接层的输入和输出通道分别对应样本和特征的二维张量。NiN在每个像素位置（高和宽）应用一个全连接层，如果将权重连接到每个空间位置，开源视其为1×1卷积层，或作为在每个像素位置上毒理作用的全连接层。空间维度中的每个像素视为单个样本，将通道维度视为不同特征。


```python
import torch
from torch import nn
from d2l import torch as d2l

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=strides, padding=padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
    )
```

##### 7.3.2 NiN模型
NiN使用窗口形状为11×11、5×5、3×3的卷积层，输出通道数与AlexNet中的相同，每个NiN块后有一个最大汇聚层，形状为3×3，步幅为2。NiN取消全连接层，取而代之的是使用一个NiN块，输出通道数等于标签类别数。最后放一个全局平均汇聚层（Global Average Pooling Layer），生成一个对数几率。NiN可以显著减少模型所需参数的数量，但可能会增加训练模型的时间。


```python
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()    # 将四维输出转成二维输出，形状为（批量大小,10）
)
```


```python
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
```

    Sequential output shape:	 torch.Size([1, 96, 54, 54])
    MaxPool2d output shape:	 torch.Size([1, 96, 26, 26])
    Sequential output shape:	 torch.Size([1, 256, 26, 26])
    MaxPool2d output shape:	 torch.Size([1, 256, 12, 12])
    Sequential output shape:	 torch.Size([1, 384, 12, 12])
    MaxPool2d output shape:	 torch.Size([1, 384, 5, 5])
    Dropout output shape:	 torch.Size([1, 384, 5, 5])
    Sequential output shape:	 torch.Size([1, 10, 5, 5])
    AdaptiveAvgPool2d output shape:	 torch.Size([1, 10, 1, 1])
    Flatten output shape:	 torch.Size([1, 10])


##### 7.3.3 训练模型


```python
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, torch.device('mps'))
```

    loss 0.333, train acc 0.876, test acc 0.873
    1929.4 examples/sec on mps



    
![svg](Chapter7_%E7%8E%B0%E4%BB%A3%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%85%B6%E4%BA%8C_files/Chapter7_%E7%8E%B0%E4%BB%A3%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%85%B6%E4%BA%8C_6_1.svg)
    



```python
lr, num_epochs, batch_size = 0.05, 15, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, torch.device('cuda:0'))
```

    loss 0.328, train acc 0.881, test acc 0.883
    841.9 examples/sec on cuda:0



    
![svg](Chapter7_%E7%8E%B0%E4%BB%A3%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%85%B6%E4%BA%8C_files/Chapter7_%E7%8E%B0%E4%BB%A3%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%85%B6%E4%BA%8C_7_1.svg)
    


#### 7.4 含并行连接的网络（GoogLeNet）
GoogLeNet吸收了NiN中串联网络的思想，并在此基础上做出改进，解决了卷积核大小适配的问题。
##### 7.4.1 Inception块
GoogLeNet中基本的卷积块为Inception块，由4条并行路径组成，前3条路径使用窗口为1×1、3×3、5×5的卷积层，从不同空间大小中提取信息。中间的2条路径在输入上执行1×1卷积，以减少通道数，降低模型复杂度。第4条路径使用3×3最大汇聚层，然后使用1×1卷积层来改变通道数。这4条路径都是用合适的填充以使输入与输出的高度和宽度一致，最后将每条路径的输出在通道维度上合并，构成Inception块输出。

**设计思路**：
不明确3×3还是5×5卷积块更有效，就让网络自己学。为了避免计算量爆炸，引入1×1卷积块降维，把计算成本控制在合理范围。各路径的通道数比例出自经验判断。


```python
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

class Inception(nn.Module):
    # c1~c4是每条路径的输出通道
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 路径1：单1×1网络
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 路径2：1×1卷积层后接3×3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 路径3：1×1卷积层后接5×5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 路径4：3×3最大汇聚曾后接1×1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
    
    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)
```

##### 7.4.2 GoogLeNet模型
GoogLeNet模型使用9个Inception块和全局平均汇聚层的堆叠来生成其估计值。Inception块之间可最大降低维度。
- 第一个模块使用64个通道，7×7卷积层；
- 第二个模块使用两个卷积层，第一个卷积层使用64个通道，1×1卷积层，第二个卷积层使用3×3卷积层；
- 第三个模块串联两个完整的Inception块，第一个Inception块的输出通道数为64+128+32+32=256，4调路径的输出通道数之比为2:4:1:1，第二条和第三条路径首先将输出通道先减少1/2和1/12，然后连接第二个卷积层。第二个Incepion块将输出通道数之比为4:6:3:2；
- 第四个模块串联5个Inception块，输出通道数分别为512、512、512、528、832；
- 第五个模块包含输出通道数为832和1024的两个Inception块，后面紧跟输出层。


```python
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (32, 128), 128),
                   nn.AdaptiveAvgPool2d((1, 1)),
                   nn.Flatten())
net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
```


```python
X = torch.rand(size=(1, 1, 96, 96))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
```

    Sequential output shape:	 torch.Size([1, 64, 24, 24])
    Sequential output shape:	 torch.Size([1, 192, 12, 12])
    Sequential output shape:	 torch.Size([1, 480, 6, 6])
    Sequential output shape:	 torch.Size([1, 832, 3, 3])
    Sequential output shape:	 torch.Size([1, 1024])
    Linear output shape:	 torch.Size([1, 10])


##### 7.4.3 训练模型


```python
lr, num_epochs, batch_size = 0.1, 20, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, torch.device('mps'))
```

    loss 0.169, train acc 0.935, test acc 0.910
    3272.7 examples/sec on mps



    
![svg](Chapter7_%E7%8E%B0%E4%BB%A3%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%85%B6%E4%BA%8C_files/Chapter7_%E7%8E%B0%E4%BB%A3%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%85%B6%E4%BA%8C_14_1.svg)
    

