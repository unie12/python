##################################
# Early Stopping
# Epoch마다 검증 dataset의 loss를 기록
# 이를 기반으로 최적의 모델을 저장하고 학습을 조기 종료
# 1. 모델의 성능을 평가하기 위한 검증 dataset 선정
# 2. 최적의 성능을 가진 모델을 저장하기 위한 변수 초기화
# 3. 학습을 중지하기 위한 조기 종료 조건 설정
# 4. 학습 과정에서 검증 데이터셋의 loss 기록
# 5. 검증 데이터셋의 손실이 이전보다 증가하는 경우, 조기 종료 조건 확인 후 학습 중지
#
# 장점
# overfitting 방지, 시간 및 자원 절약
# 단점
# 조기 종료를 할 때, 모델이 충분한 학습을 할 수 없을 수도 있음
####################################

import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# model 초기화
model = Net()

# loss function 초기화
criterion = nn.CrossEntropyLoss()

# optimizer 초기화 (Stochastic Gradient Descent)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 조기 종료 변수 초기화
early_stopping_epchs = 5
best_loss = float('inf')
early_stop_counter = 0

# Loop
for epoch in range(100):
    # 학습
    model.train()
    train_loss = 0.0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
    # 검증
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for data, target in valid_loader:
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)
    # 검증 데이터셋의 손실이 이전보다 증가하는 경우
    if valid_loss > best_loss:
        early_stop_counter += 1
    else:
        best_loss = valid_loss
        early_stop_counter = 0
    
    # 조기 종료 조건 확인
    if early_stop_counter >= early_stopping_epchs:
        print("Early Stopping!")
        break