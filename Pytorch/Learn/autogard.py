import torch
from torchvision.models import resnet18, ResNet18_Weights ## resnet18 모델 불러오기
model = resnet18(weights = ResNet18_Weights.DEFAULT)
## 3채널짜리 높이와 넓이가 64인 이미지 하나를 표현하는 무작위 데이터 텐서 생성, label 무작위 초기화
## 정답(label)은 (1,1000)의 모양
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

## forward pass
prediction = model(data)

## backward pass
loss = (prediction - labels).sum()
# print(loss)
loss.backward()

## gradient descent
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step()
# print(optim)