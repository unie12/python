import torch
import numpy as np

## 데이터로부터 직접 텐서 생성
data = [[1,2], [3, 4]]
x_data = torch.tensor(data)

## numpy배열로부터 텐서 생성
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

## 다른 텐서로부터 텐서 생성
x_ones = torch.ones_like(x_data)
# print(f"Ones Tensor: \n {x_ones} \n")
x_rand = torch.rand_like(x_data, dtype=torch.float)
# print(f"Random Tensor: \n {x_rand} \n")

## random or constant 값 사용
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

# print(f"Random Tensor: \n {rand_tensor} \n")
# print(f"Ones Tensor: \n {ones_tensor} \n")
# print(f"Zeros Tensor: \n {zeros_tensor} \n")

## 텐서의 attribute
tensor = torch.rand(3, 4)

# print(f"Shape of tensor: {tensor.shape}")
# print(f"Datatype of tensor: {tensor.dtype}")
# print(f"Dvice tensor is stored on: {tensor.device}")

## 텐서 연산
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    # print(f"Device tesnor is sotred on: {tensor.device}")

tensor = torch.ones(4, 4)
tensor[:, 1] = 0
# print(tensor)

t1 = torch.cat([tensor, tensor, tensor] , dim=1)
# print(t1)

# print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# print(f"tensor * tensor \n {tensor * tensor}")

# print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

# print(tensor, "\n")
tensor.add_(5)
# print(tensor)

t = torch.ones(5)
# print(f"t: {t}")
n = t.numpy()
# print(f"n: {n}")

t.add_(1)
# print(f"t: {t}")
# print(f"n: {n}")

n = np.ones(5)
t = torch.from_numpy(n)
np.add(n,  1, out=n)
# print(f"t: {t}")
# print(f"n: {n}")

