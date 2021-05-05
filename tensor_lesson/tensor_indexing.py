import torch


batch_size = 10
features = 25
x = torch.rand((batch_size, features))

print(x.shape)
print(x[0].shape)  # x[0,:]
print(x[:, 0].shape)  # 0:10 -> [0, 1, 2, ..., 9]

x[0, 0] = 100

# 中級向け
x = torch.arange(10)
indices = [2, 5, 8]
print(x)
print(x[indices])

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x)
print(rows)
print(cols)
print(x[rows, cols])

# 上級向け
x = torch.arange(10)
# xが２より下かxが８より上か
print(x[(x < 2) | (x > 8)])
# 偶数のみ
print(x[x.remainder(2) == 0])

print(torch.where(x > 5, x, x*2))
print(torch.tensor([0, 0, 1, 2, 2, 3, 4]).unique())
print(x.ndimension())
# 要素数を取得
print(x.numel())