import torch


# GPUかCPUかの設定
device = "cuda" if torch.cuda.is_available() else "cpu"


# ======================== #
#      よく使うmethod       #
# ======================== #

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32,
                         device=device, requires_grad=True)
print(my_tensor)
print('dtype: ', my_tensor.dtype)
print('device: ', my_tensor.device)
print('shape: ', my_tensor.shape)
print('requires grad: ', my_tensor.requires_grad)

# Other common initialization methods
# 0だけの行列作成
x = torch.empty(size=(3, 3))
x = torch.zeros((3, 3))

# ランダム生成
x = torch.rand((3, 3))

# 1だけを作成
x = torch.ones((3, 3))

# 単位行列
x = torch.eye(5, 5)

# startからendまで順に作成
x = torch.arange(start=0, end=5, step=1)
x = torch.linspace(start=0.1, end=1, steps=10)

# 正規化した行列
x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
x = torch.empty(size=(1, 5)).uniform_(0, 1)

x = torch.diag(torch.ones(3))
