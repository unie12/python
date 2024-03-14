import torch
# FashionMNIST 데이터셋 불러오기
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import os
import pandas as pd
from torchvision.io import read_image

from torch.utils.data import DataLoader

# root -> 데이터가 저장되는 경로
# train -> 학습용 또는 테스트용 데이터셋 여부 지정
# download = true -> root에 데이터가 없는 경우 인터넷에서 다운로드
# transform, target_tranform -> feature과 label 변형을 지정
 
# dataset 불러오기
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# dataset 순회하고 시각화하기
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8,8))
cols, rows = 3,3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()


## 파일에서 사용자 정의 데이터셋 만들기
# 반드시 3개 함수 구현 "__init__", "__len__", "__getitem__"
class CustomImageDataset(Dataset):
    # dataset 객체가 생성될 떄 한 번만 실행
    # 이미지와 주석파일이 포함된 디렉토리와 두가지 transform 초기화
    def __init__(self, annotations_file, img_dir, transform = None, target_tranform=None):
        self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_tranform

    # 데이터셋의 샘플 개수 반환
    def __len__(self):
        return len(self.img_labels)

    # 주어진 인덱스에 해당하는 샘플을 데이터셋에서 불러오고 반환
    def __getitem__(self,idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx,1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

## DataLoader로 학습용 데이터 준비하기
# dataset은 데이터셋의 특징을 가져오고 하나의 샘플에 label을 지정하는 일을 한 번에 함
# 모델을 학습할 때 일반적으로 샘플들을 미니배치로 전달하고, 매 epoch마다 데이터를 다시 섞어서
# overfitting을 막고, multiprocesing의 사용하여 데이터 검색 속도를 높이고자 함
# dataloader = 이러한 복잡한 과정들을 추상화한 iterable한 객체

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# dataloader를 통해 순회하기
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape:  {train_features.size()}")
print(f"Labels batch shape:  {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")