import torch


x = torch.arange(9)

# nxn に変更
x_3x3 = x.view(3, 3)
x_3x3 = x.reshape(3, 3)

print(x_3x3)

# 転値
y = x_3x3.t()
print(y)

# contiguousを使うことで転置した行列を変換できる
print(y.contiguous().view(9))

x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
# 行で結合
print(torch.cat((x1, x2), dim=0))
print(torch.cat((x1, x2), dim=0).shape)
# 列で結合
print(torch.cat((x1, x2), dim=1))
print(torch.cat((x1, x2), dim=1).shape)

# ベクトルに変換
z = x1.view(-1)
print(z)
print(x1.shape, z.shape)

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)
print(z.shape)

z = x.permute(0, 2, 1)
print(z.shape)

# 次元を増やす
x = torch.arange(10)
print(x.shape)
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)

x = torch.arange(10).unsqueeze(0).unsqueeze(1)  # 1x1x10
print(x.shape)
