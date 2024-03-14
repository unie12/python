import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
## 간단한 신경망 모델 정의, 초기화하는 예제

## 사용 가능한 디바이스 확인 후 모델을 해당 디바이스에 할당
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# print(f"Using {device} device")

## 신경망 모델을 nn.Module의 하위클래스로 정의
class NeuralNetwork(nn.Module):
    # 신경망 계층들 초기화(flatten 평탄화)
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    # 입력 데이터에 대한 연산들 구현
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)
# print(model)

## 무작위 데이터를 생성하여 모델을 통과시킨 후, 예측된 클래스 출력
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
# print(f"Predicated class: {y_pred}")

## 28*28 크기의 이미지 3개로 구성된 미니배치
input_image = torch.rand(3, 28, 28)
print(input_image.size())

## nn.Flatten 계층 초기화하여 28*28의 2d 이미지를 784 픽셀 값을 갖는 연속 배열로 변환
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

## 선형 계층은 저장된 weight와 bias를 사용하여 입력에 선형 변환을 적용
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

## nn.ReLU: 비선형 활성화는 선형 변환 후에 적용되어 비선형성 도입하여 신경망이 다양한 현상 학습
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

## nn.Sequential: 순서를 갖는 모듈의 컨테이너
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)

## nn.softmax: 
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

## 신경망 내부의 많은 layers들은 parameterize됨
## 학습 중에 최적화되는 weight과 bias와 연관
## nn.Module 상속시 모델 객체 내부의 모든 필드들이 자동으로 track, 모든 매개변수에 접근 가능
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n")