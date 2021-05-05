import torch.nn as nn
import torch.nn.functional as F


class NN(nn.Module):
    """
    シンプルなNNモデルクラス
    """
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        """
        順伝播

        :param x: 目的変数
        :return: 順伝播後のx(64 x num_class)
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
