import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# transform을 통해 데이터를 학습에 필요한 형태로 조작

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
# print(ds)