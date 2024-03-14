import numpy as np
import torch

## 데이터로부터 직접 텐서 생성
data = [[1,2], [3,4]]
x_data = torch.tensor(data)
# print(x_data)

## numpy 배열로부터 생성
np_array = np.array(data)
# print(np_array)
x_np = torch.from_numpy(np_array)
# print(x_np)

## 다른 텐서로부터 생성
x_ones = torch.ones_like(x_data)
# print(f"Ones Tensor : \n {x_ones} \n")
x_rand = torch.rand_like(x_data, dtype=torch.float)
# print(f"Random Tensor : \n {x_rand} \n")

## 무작위 또는 상수값 사용
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

# print(f"Random Tensor: \n {rand_tensor} \n")
# print(f"Ones Tensor: \n {ones_tensor} \n")
# print(f"Zero Tensor: \n {zeros_tensor} \n")

## 텐서의 속성
tensor = torch.rand(3,4)
# print(f"Shape of tensor: {tensor.shape}")
# print(f"Datatype of tensor: {tensor.dtype}")
# print(f"Device tensor is stored on: {tensor.device}")

## 텐서 연산
## .to 메서드를 사용하면 GPU로 텐서를 명시적으로 이동
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

## numpy식의 표준 인덱싱과 슬라이싱
tensor = torch.ones(4,4)
# print(f"First row: {tensor[0]}")
# print(f"First column: {tensor[:, 0]}")
# print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
# print(tensor)

## 텐서 합치기
t1 = torch.cat([tensor, tensor, tensor], dim=1)
# print(t1)

## 산술연산
# 행렬곱 게산
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
# print(tensor)

# 요소별 곱 계산
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
# print(tensor)

# 단일 요소 텐서
agg = tensor.sum()
agg_item = agg.item()
# print(agg_item, type(agg_item))

# 바꿔치기 연산 -> 메모리를 일부 절약지만, history가 즉시 삭제되어 사용 권장x
# print(f"{tensor} \n")
tensor.add_(5)
# print(tensor)

## Numpy 변환(bridge)
# tensor -> numpy array
# cpu 상의 텐서와 numpy 배열은 메모리 공간 공유 -> 하나 변경 시 다른 하나도 변경
t = torch.ones(5)
# print(f"t: {t}")
n = t.numpy()
# print(f"n: {n}")
t.add_(1)
# print(f"t : {t}")
# print(f"n : {n}")

# numpy array -> tensor
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n,1,out=n)
# print(f"t: {t}")
# print(f"n: {n}")