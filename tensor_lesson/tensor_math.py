import torch


x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# 足し算
z1 = torch.empty(3)
torch.add(x, y, out=z1)

z2 = torch.add(x, y)
z = x + y

# 引き算
z = x - y

# 割り算
z = torch.true_divide(x, y)

t = torch.zeros(3)
t.add(x)
t += x  # t = t + x

# 指数関数
z = x.pow(2)
z = x ** 2

# 比較
z = x > 0
z = x < 0

# 行列の比較
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)
x3 = x1.mm(x2)

# 行列の指数
matrix_exp = torch.rand(5, 5)
print(matrix_exp)
print(matrix_exp.matrix_power(3))

# 掛け算
z = x * y
print(z)

# dot
z = torch.dot(x, y)
print(z)

# バッチごとに2次元×2次元の行列積
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2)

print(out_bmm)

# Broadcasting
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))

z = x1 - x2
z = x1 ** x2

# 足し算
sum_x = torch.sum(x, dim=0)

# 最小値、最大値
values, indices = torch.max(x, dim=0)
values, indices = torch.min(x, dim=0)

# 絶対値
abs_x = torch.abs(x)

z = torch.argmax(x, dim=0)
z = torch.argmin(x.float(), dim=0)

# 平均値
mean_x = torch.mean(x.float(), dim=0)

# 平均値
z = torch.eq(x, y)

# sort
sorted_y, indices = torch.sort(y, dim=0, descending=False)

# 値の制限
z = torch.clamp(x, min=0)

x = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)
z = torch.any(x)
z = torch.all(x)
