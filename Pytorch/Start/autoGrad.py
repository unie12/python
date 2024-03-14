import torch

## 입력 x 매개변수 w b, 손실함수
## w b는 최적화를 해야하는 매개변수
x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z,y)

# print(f"Gradient function for z = {z.grad_fn}")
# print(f"Gradient function for loss = {loss.grad_fn}")

## Gradient 계산
loss.backward()
# print(w.grad)
# print(b.grad)

## Gradient tracking 멈추기
## requires_grad=True인 모든 텐서들은 연산 기록을 추적하고 변화도 계사능ㄹ 지원
## 모델 학습 뒤 입력 데이터를 단순 적용하는 경우에는 이런 지원이 필요 없음
## 또한 tracking을 멈춰야 하는 이유는 고정된 매개변수로 표시, 연산 더 효율적 -> 속도 향상
z = torch.matmul(x, w) + b
# print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w) + b
# print(z.requires_grad)

z = torch.matmul(x, w) + b
z_det = z.detach()
# print(z_det.requires_grad)

## optional reading -> Jacobian product 계산(스칼라 손실 함수를 가지고 일부 매개변수와 관련 변화도를 계산해야 할 때)
## gradient가 누적
inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call]n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"Second call]n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")

