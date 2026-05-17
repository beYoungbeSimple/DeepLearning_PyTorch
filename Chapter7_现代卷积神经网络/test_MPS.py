import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from d2l import torch as d2l

# =========================================================
# 1. MPS设备
# =========================================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

# =========================================================
# 2. 超参数（针对 Mac Studio + MPS 优化）
# =========================================================
batch_size = 512          # 不建议4096，MPS通常512~1024效率更高
lr = 0.005                # 大batch对应更大学习率
num_epochs = 20
num_workers = 8           # 提高CPU数据预处理并行度

# =========================================================
# 3. 数据集
# =========================================================
# FashionMNIST 原始只有28x28
# 如果你用AlexNet/ResNet，96已经足够
# 224纯属浪费显存和带宽

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])

train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_iter = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    persistent_workers=True
)

test_iter = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    persistent_workers=True
)

# =========================================================
# 4. 网络（AlexNet简化版）
# =========================================================
net = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(128, 256, kernel_size=3, padding=1),
    nn.ReLU(),

    nn.Flatten(),

    nn.Linear(256 * 24 * 24, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),

    nn.Linear(1024, 10)
)

# =========================================================
# 5. channels_last（Apple Silicon优化）
# =========================================================
net = net.to(device, memory_format=torch.channels_last)

# =========================================================
# 6. PyTorch 2 编译优化
# =========================================================
net = torch.compile(net)

# =========================================================
# 7. 损失函数与优化器
# =========================================================
loss = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    net.parameters(),
    lr=lr,
    weight_decay=1e-4
)

# =========================================================
# 8. 学习率调度（Cosine）
# =========================================================
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs
)

# =========================================================
# 9. 训练
# =========================================================
for epoch in range(num_epochs):

    net.train()

    metric_loss = 0
    metric_acc = 0
    metric_num = 0

    for X, y in train_iter:

        X = X.to(
            device,
            memory_format=torch.channels_last,
            non_blocking=True
        )

        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        y_hat = net(X)

        l = loss(y_hat, y)

        l.backward()

        optimizer.step()

        metric_loss += l.item() * y.shape[0]
        metric_acc += (y_hat.argmax(dim=1) == y).sum().item()
        metric_num += y.shape[0]

    scheduler.step()

    train_loss = metric_loss / metric_num
    train_acc = metric_acc / metric_num

    # =====================================================
    # 测试
    # =====================================================
    net.eval()

    test_acc = 0
    test_num = 0

    with torch.no_grad():

        for X, y in test_iter:

            X = X.to(
                device,
                memory_format=torch.channels_last,
                non_blocking=True
            )

            y = y.to(device, non_blocking=True)

            y_hat = net(X)

            test_acc += (y_hat.argmax(dim=1) == y).sum().item()
            test_num += y.shape[0]

    test_acc /= test_num

    # =====================================================
    # MPS显存查看
    # =====================================================
    memory_gb = torch.mps.current_allocated_memory() / 1024**3

    print(
        f"Epoch {epoch+1:2d} | "
        f"train loss {train_loss:.4f} | "
        f"train acc {train_acc:.4f} | "
        f"test acc {test_acc:.4f} | "
        f"lr {scheduler.get_last_lr()[0]:.6f} | "
        f"MPS memory {memory_gb:.2f} GB"
    )