import torch
from torch.utils.data import Dataset, DataLoader

print('=== EP01 环境自检 ===')
print('torch 版本:', torch.__version__)
print('CUDA 可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU 名称:', torch.cuda.get_device_name(0))
    print('CUDA 版本:', torch.version.cuda)
else:
    print('当前走 CPU，本教程也能继续学。')

print('\n=== EP02 Tensor 基本操作 ===')
x = torch.tensor([[1, 2], [3, 4]])
print(x)
print(x.shape)
print(x.dtype)
print(x.device)
zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)
randn = torch.randn(2, 3)
arange = torch.arange(0, 10)
print(zeros.shape, ones.shape, randn.shape, arange.shape)
x2 = torch.arange(12)
x2 = x2.reshape(3, 4)
x2 = x2.view(2, 6)
x2 = x2.unsqueeze(0).squeeze(0)
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])
print(a + b)
print(torch.sum(a), torch.mean(a))

print('\n=== EP03 自动求导 ===')
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print('x.grad =', x.grad)
w = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (w ** 2).sum()
loss.backward()
print('w.grad =', w.grad)
w2 = torch.tensor(1.0, requires_grad=True)
for i in range(3):
    loss = w2 ** 2
    loss.backward()
    print('loop grad', i, w2.grad.item())
    w2.grad.zero_()

print('\n=== EP04 nn.Module ===')
import torch.nn as nn
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
model = SimpleNet()
out = model(torch.randn(8, 10))
print('SimpleNet out shape:', out.shape)
print('params:', sum(p.numel() for p in model.parameters()))

print('\n=== EP05 Dataset 与 DataLoader ===')
class ToyDataset(Dataset):
    def __init__(self):
        self.x = torch.randn(100, 10)
        self.y = torch.randint(0, 2, (100,))
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
dataset = ToyDataset()
loader = DataLoader(dataset, batch_size=16, shuffle=True)
print('dataset len:', len(dataset))
first = dataset[0]
print('sample shapes:', first[0].shape, first[1].shape if hasattr(first[1], 'shape') else 'scalar')
for batch_x, batch_y in loader:
    print('batch shapes:', batch_x.shape, batch_y.shape)
    break

print('\n基础 demo 验证完成')
