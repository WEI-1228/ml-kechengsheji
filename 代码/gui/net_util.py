import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
fm_net = models.resnet18()
num_ftrs = fm_net.fc.in_features
fm_net.fc = nn.Linear(num_ftrs, 2)
fm_net.load_state_dict(torch.load('model/FM_model.pt'))
fm_net = fm_net.to(device)
fm_net.eval()
fm_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def detect_sex(img):
    ans = ['female', 'male']
    img = fm_transform(Image.fromarray(img)).reshape(1, 3, 224, 224).to(device)
    with torch.no_grad():
        fm_net.eval()
        res = fm_net(img)
    out = res.detach().cpu().numpy().argmax()
    return ans[out]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 4, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(5 * 5 * 64, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 5 * 5 * 64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


emotion_net = Net()
emotion_net.load_state_dict(torch.load('model/net_params.pt'))  # 仅加载参数
emotion_net = emotion_net.to(device)

emotion_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(42),
    transforms.ToTensor(),
])


def detect_emotion(img):
    ans = ['happy', 'normal', 'sad']
    img = emotion_transform(Image.fromarray(img)).reshape(1, 1, 42, 42).to(device)
    with torch.no_grad():
        emotion_net.eval()
        res = emotion_net(img)
    out = res.detach().cpu().numpy().argmax()
    return ans[out]
