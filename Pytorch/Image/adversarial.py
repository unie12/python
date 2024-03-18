########################################################################
# Adversarial Example Generation
# 의도적으로 모델을 속이는 방식
# 원본 이미지에 노이즈를 섞음으로써 사람의 눈으로는 분간할 수 없지만
# 실제 모델이 입력받는 데이터 형식차가 발생하여 성능이 바뀜
# 화이트박스(배경 정보 아는 상태) - 블랙박스(모르는 상태) 방식
# 화이트박스 방식 중 유명한 공격: FGSM -> Gradients를 활용하여 공격
# -> 손실을 최소화하는 것이 아니라 손실을 최대화하는 방향으로 입력 데이터 조정
########################################################################

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# 일종의 학습률 개념, 노이즈가 너무 커지지 않고 사람 눈에 보이지 않게 제한
epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = "Pytorch\Image\lenet_mnist_model.pth"
use_cuda = True

# 공격을 받는 모델
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# MNIST 테스트 데이터셋과 데이터로더 선언
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=1, shuffle=True)

print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

model = Net().to(device)
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
model.eval()

## FGSM 공격
## FGSM(원본 이미지, 픽셀 단위의 작은 변화 주는 값, 입력 영상에 대한 변화도 손실 값)
def fgsm_attack(image, epsilon, data_grad):
    # data_grad의 요소별 부호 값 얻어오기
    sign_data_grad = data_grad.sign()
    # 입력 이미지의 각 픽셀에 sign_data_grad를 적용해 작은 변화가 적용된 이미지 생성
    perturbed_image = image + epsilon * sign_data_grad
    # 값 범위[0,1] 유지하기 위해 clipping
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

## Testing 함수
def test(model, device, test_loader, epsilon):

    correct = 0
    adv_examples = []

    # 테스트 셋의 모든 예제에 대해 루프
    for data, target in test_loader:
        # 디바이스(cpu or gpu)에 데이터와 라벨 값 보내기
        data, target = data.to(device), target.to(device)
        # 텐서의 속성 중 required_grad 설정 -> 공격에서 중요한 부분?
        # gradient 계산 포함 여부를 물어보는 건데 True하면 backward를 통해 연산 추적해서 미분
        data.requires_grad = True

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # 로그 확률의 최댓값 가지는 인덱스 얻기

        # 만약 초기 예측이 틀리면, 공격하지 않도록 하고 계속 진행
        if init_pred.item() != target.item():
            continue

        ## backward -> FGSM 공격
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # 공격 후 재분류
        output = model(perturbed_data)

        # 올바른지 확인
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
            # 0 epsilon 예제에 대해 저장 (squeeze = 텐서에서 차원이 1인 부분 제거)
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # 추후 시각화를 위해 다른 예제들 저장
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # 해당 epsilon에서의 최종 정확도 계산
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    return final_acc, adv_examples
        

## 공격 실행
## epsilon 값이 커질수록 accuray 낮아짐
accuracies = []
examples = []

for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

## epsilon이 선형적으로 분포하더라도 accuray는 곡선의 추세
plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()