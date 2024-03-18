import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module): ## nn.Module = 신경망 모듈, 매개변수를 encapulation하는 방법

    def __init__(self):
        super(Net, self).__init__()
        ## 입력 채널 1개, 출력 채널 6개, 5*5의 컨볼루션 행렬
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        ## affine 연산: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 *5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        ## (2, 2) 크기 윈도우에 대해 max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        ## 크기가 제곱수라면, 하나의 숫자만의 specify
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # batch 차원을 제외한 모든 차원을 하나로 평탄화
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
net = Net()
# print(net)

params = list(net.parameters())
# print(len(params))
# print(params[0].size()) ## conv1의 weight

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

net.zero_grad()
out.backward(torch.randn(1, 10))

output = net(input)
target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
# print(loss)

# print(loss.grad_fn) # MSELoss
# print(loss.grad_fn.next_functions[0][0]) # Linear
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) # ReLU

net.zero_grad()
print('conv1.bias.grad before backwad')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)