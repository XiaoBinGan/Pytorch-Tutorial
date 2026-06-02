import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image, ImageDraw

BASE_DIR = os.path.dirname(__file__)
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
SAMPLES_DIR = os.path.join(BASE_DIR, 'samples')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device =', device)

class ToyDataset(Dataset):
    def __init__(self, n=200, in_dim=10, num_classes=2):
        self.x = torch.randn(n, in_dim)
        self.y = torch.randint(0, num_classes, (n,))
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.net(x)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class MNISTNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

print('\n=== EP06 完整训练循环 ===')
dataset = ToyDataset()
loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
losses = []
for epoch in range(3):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg = total_loss / len(loader)
    losses.append(avg)
    print(f'epoch={epoch+1}, loss={avg:.4f}')

print('\n=== EP08 CNN shape 验证 ===')
cnn = SmallCNN()
out = cnn(torch.randn(4, 1, 28, 28))
print('cnn out shape:', out.shape)

print('\n=== EP09 验证集准确率 ===')
full_dataset = ToyDataset(n=240)
train_dataset, val_dataset = random_split(full_dataset, [200, 40])
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
print('val acc =', correct / total)

print('\n=== EP10 保存 / 加载 / 断点续训 ===')
mnist_model = MNISTNet().to(device)
optimizer2 = torch.optim.Adam(mnist_model.parameters(), lr=1e-3)
state_path = os.path.join(CHECKPOINT_DIR, 'mnist_model.pt')
ckpt_path = os.path.join(CHECKPOINT_DIR, 'last_checkpoint.pt')
torch.save(mnist_model.state_dict(), state_path)
sd = torch.load(state_path, map_location='cpu')
model2 = MNISTNet()
model2.load_state_dict(sd)
model2.eval()
dummy = torch.randn(2, 1, 28, 28)
loss = nn.CrossEntropyLoss()(mnist_model(dummy.to(device)), torch.tensor([1, 2], device=device))
torch.save({'epoch': 0, 'model_state_dict': mnist_model.state_dict(), 'optimizer_state_dict': optimizer2.state_dict(), 'loss': loss.item()}, ckpt_path)
ckpt = torch.load(ckpt_path, map_location='cpu')
model2.load_state_dict(ckpt['model_state_dict'])
print('checkpoint keys:', sorted(ckpt.keys()))

print('\n=== EP12 设备与混合精度结构验证 ===')
amp_model = SimpleNet().to(device)
amp_optimizer = torch.optim.Adam(amp_model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
for x, y in DataLoader(ToyDataset(), batch_size=8):
    x, y = x.to(device), y.to(device)
    amp_optimizer.zero_grad()
    with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
        logits = amp_model(x)
        loss = criterion(logits, y)
    scaler.scale(loss).backward()
    scaler.step(amp_optimizer)
    scaler.update()
    print('amp one-step ok, loss=', float(loss.item()))
    break

print('\n=== EP13 推理脚本链路 ===')
img_path = os.path.join(SAMPLES_DIR, 'sample.png')
img = Image.new('L', (28, 28), color=0)
d = ImageDraw.Draw(img)
d.text((8, 4), '3', fill=255)
img.save(img_path)
img = Image.open(img_path)
x = torch.tensor(list(img.getdata()), dtype=torch.float32).reshape(1, 1, 28, 28) / 255.0
with torch.no_grad():
    pred = torch.argmax(model2(x), dim=1).item()
print('single image predict ok, pred =', pred)

print('\n=== EP15 模板文件最小落地 ===')
project_demo = os.path.join(BASE_DIR, 'project_demo')
os.makedirs(project_demo, exist_ok=True)
for name in ['data', 'checkpoints', 'outputs']:
    os.makedirs(os.path.join(project_demo, name), exist_ok=True)
files = {
    'config.py': "DATA_DIR='data'\nCHECKPOINT_DIR='checkpoints'\nOUTPUT_DIR='outputs'\nBATCH_SIZE=64\nLR=1e-3\nEPOCHS=1\nNUM_CLASSES=10\n",
    'models.py': "import torch.nn as nn\n\nclass MNISTNet(nn.Module):\n    def __init__(self, num_classes=10):\n        super().__init__()\n        self.net = nn.Sequential(\n            nn.Flatten(),\n            nn.Linear(28 * 28, 128),\n            nn.ReLU(),\n            nn.Linear(128, num_classes)\n        )\n    def forward(self, x):\n        return self.net(x)\n",
    'infer.py': "import sys, torch\nfrom PIL import Image\nfrom torchvision import transforms\nfrom models import MNISTNet\nmodel = MNISTNet()\nstate_dict = torch.load('checkpoints/mnist_model.pt', map_location='cpu')\nmodel.load_state_dict(state_dict)\nmodel.eval()\ntransform = transforms.Compose([transforms.Grayscale(), transforms.Resize((28, 28)), transforms.ToTensor()])\nimage = Image.open(sys.argv[1])\nx = transform(image).unsqueeze(0)\nwith torch.no_grad():\n    pred = torch.argmax(model(x), dim=1).item()\nprint('预测结果:', pred)\n"
}
for name, content in files.items():
    with open(os.path.join(project_demo, name), 'w', encoding='utf-8') as f:
        f.write(content)
# create checkpoint for infer.py
state_target = os.path.join(project_demo, 'checkpoints', 'mnist_model.pt')
torch.save(MNISTNet().state_dict(), state_target)
# create sample image
sample_target = os.path.join(project_demo, 'sample.png')
Image.new('L', (28, 28), color=255).save(sample_target)
# run infer.py as smoke test
import subprocess
proc = subprocess.run([sys.executable, os.path.join(project_demo, 'infer.py'), sample_target], cwd=project_demo, capture_output=True, text=True)
print('infer return code =', proc.returncode)
print(proc.stdout.strip())
if proc.returncode != 0:
    print(proc.stderr)
    raise SystemExit(proc.returncode)

print('\n后半段 demo 验证完成')
