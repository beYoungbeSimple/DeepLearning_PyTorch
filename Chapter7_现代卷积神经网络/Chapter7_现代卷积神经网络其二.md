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
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, torch.device('cuda:0'))
```

    loss 0.358, train acc 0.867, test acc 0.851
    1156.3 examples/sec on cuda:0



    
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


```python
from torch.nn import functional as F

class Inception(nn.Module):
    # c1~c4是每条路径的输出通道
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 路径1：单1×1网络

```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Cell In[1], line 1
    ----> 1 from torch.nn import functional as F
          3 class Inception(nn.Module):
          4     # c1~c4是每条路径的输出通道
          5     def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):


    File E:\Anaconda\envs\d2l\lib\site-packages\torch\__init__.py:2229
       2222 from torch._compile import _disable_dynamo  # usort: skip
       2224 ################################################################################
       2225 # Import interface functions defined in Python
       2226 ################################################################################
       2227 
       2228 # needs to be after the above ATen bindings so we can overwrite from Python side
    -> 2229 from torch import _VF as _VF, functional as functional  # usort: skip
       2230 from torch.functional import *  # usort: skip # noqa: F403
       2232 ################################################################################
       2233 # Remove unnecessary members
       2234 ################################################################################


    File E:\Anaconda\envs\d2l\lib\site-packages\torch\functional.py:8
          5 from typing import Any, TYPE_CHECKING
          7 import torch
    ----> 8 import torch.nn.functional as F
          9 from torch import _VF, Tensor
         10 from torch._C import _add_docstr


    File E:\Anaconda\envs\d2l\lib\site-packages\torch\nn\__init__.py:8
          1 # mypy: allow-untyped-defs
          2 from torch.nn.parameter import (  # usort: skip
          3     Buffer as Buffer,
          4     Parameter as Parameter,
          5     UninitializedBuffer as UninitializedBuffer,
          6     UninitializedParameter as UninitializedParameter,
          7 )
    ----> 8 from torch.nn.modules import *  # usort: skip # noqa: F403
          9 from torch.nn import (
         10     attention as attention,
         11     functional as functional,
       (...)
         16     utils as utils,
         17 )
         18 from torch.nn.parallel import DataParallel as DataParallel


    File E:\Anaconda\envs\d2l\lib\site-packages\torch\nn\modules\__init__.py:1
    ----> 1 from .module import Module  # usort: skip
          2 from .linear import Bilinear, Identity, LazyLinear, Linear  # usort: skip
          3 from .activation import (
          4     CELU,
          5     ELU,
       (...)
         32     Threshold,
         33 )


    File E:\Anaconda\envs\d2l\lib\site-packages\torch\nn\modules\module.py:17
         15 from torch._prims_common import DeviceLikeType
         16 from torch.nn.parameter import Buffer, Parameter
    ---> 17 from torch.utils._python_dispatch import is_traceable_wrapper_subclass
         18 from torch.utils.hooks import BackwardHook, RemovableHandle
         21 __all__ = [
         22     "register_module_forward_pre_hook",
         23     "register_module_forward_hook",
       (...)
         30     "Module",
         31 ]


    File E:\Anaconda\envs\d2l\lib\site-packages\torch\utils\__init__.py:8
          5 import weakref
          7 import torch
    ----> 8 from torch.utils import (
          9     backcompat as backcompat,
         10     collect_env as collect_env,
         11     data as data,
         12     deterministic as deterministic,
         13     hooks as hooks,
         14 )
         15 from torch.utils.backend_registration import (
         16     generate_methods_for_privateuse1_backend,
         17     rename_privateuse1_backend,
         18 )
         19 from torch.utils.cpp_backtrace import get_cpp_backtrace


    File E:\Anaconda\envs\d2l\lib\site-packages\torch\utils\data\__init__.py:1
    ----> 1 from torch.utils.data.dataloader import (
          2     _DatasetKind,
          3     DataLoader,
          4     default_collate,
          5     default_convert,
          6     get_worker_info,
          7 )
          8 from torch.utils.data.datapipes._decorator import (
          9     argument_validation,
         10     functional_datapipe,
       (...)
         14     runtime_validation_disabled,
         15 )
         16 from torch.utils.data.datapipes.datapipe import (
         17     DataChunk,
         18     DFIterDataPipe,
         19     IterDataPipe,
         20     MapDataPipe,
         21 )


    File E:\Anaconda\envs\d2l\lib\site-packages\torch\utils\data\dataloader.py:26
         24 import torch
         25 import torch.distributed as dist
    ---> 26 import torch.utils.data.graph_settings
         27 from torch._utils import ExceptionWrapper
         28 from torch.utils.data import _utils


    File E:\Anaconda\envs\d2l\lib\site-packages\torch\utils\data\graph_settings.py:8
          5 from typing_extensions import deprecated
          7 import torch
    ----> 8 from torch.utils.data.datapipes.iter.sharding import (
          9     _ShardingIterDataPipe,
         10     SHARDING_PRIORITIES,
         11 )
         12 from torch.utils.data.graph import DataPipe, DataPipeGraph, traverse_dps
         15 __all__ = [
         16     "apply_random_seed",
         17     "apply_sharding",
       (...)
         20     "get_all_graph_pipes",
         21 ]


    File E:\Anaconda\envs\d2l\lib\site-packages\torch\utils\data\datapipes\__init__.py:1
    ----> 1 from torch.utils.data.datapipes import dataframe as dataframe, iter as iter, map as map


    File E:\Anaconda\envs\d2l\lib\site-packages\torch\utils\data\datapipes\iter\__init__.py:1
    ----> 1 from torch.utils.data.datapipes.iter.callable import (
          2     CollatorIterDataPipe as Collator,
          3     MapperIterDataPipe as Mapper,
          4 )
          5 from torch.utils.data.datapipes.iter.combinatorics import (
          6     SamplerIterDataPipe as Sampler,
          7     ShufflerIterDataPipe as Shuffler,
          8 )
          9 from torch.utils.data.datapipes.iter.combining import (
         10     ConcaterIterDataPipe as Concater,
         11     DemultiplexerIterDataPipe as Demultiplexer,
       (...)
         14     ZipperIterDataPipe as Zipper,
         15 )


    File E:\Anaconda\envs\d2l\lib\site-packages\torch\utils\data\datapipes\iter\callable.py:8
          5 from typing import Any, TypeVar
          7 import torch
    ----> 8 from torch.utils.data._utils.collate import default_collate
          9 from torch.utils.data.datapipes._decorator import functional_datapipe
         10 from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper


    File E:\Anaconda\envs\d2l\lib\site-packages\torch\utils\data\_utils\__init__.py:53
         47     python_exit_status = True
         50 atexit.register(_set_python_exit_flag)
    ---> 53 from . import collate, fetch, pin_memory, signal_handling, worker


    File E:\Anaconda\envs\d2l\lib\site-packages\torch\utils\data\_utils\collate.py:330
        327 import numpy as np
        329 # For both ndarray and memmap (subclass of ndarray)
    --> 330 default_collate_fn_map[np.ndarray] = collate_numpy_array_fn
        331 # See scalars hierarchy: https://numpy.org/doc/stable/reference/arrays.scalars.html
        332 # Skip string scalars
        333 default_collate_fn_map[(np.bool_, np.number, np.object_)] = collate_numpy_scalar_fn


    AttributeError: module 'numpy' has no attribute 'ndarray'



```python

```
