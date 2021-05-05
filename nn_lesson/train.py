import torch
import torch.optim as optim
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, input_size, num_classes, learning_rate, num_epochs, train_loader):
    """
    train関数

    :param model: モデル
    :param input_size: インプットサイズ
    :param num_classes: 判定するクラス数
    :param learning_rate: 学習率
    :param num_epochs: epoch回数
    :param train_loader: trainデータセット
    :return model: モデル
    """
    model = model(input_size=input_size, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device)
            targets = targets.to(device)

            # 次元を変換
            data = data.reshape(data.shape[0], -1)

            # forward
            score = model(data)
            loss = criterion(score, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

    return model


def check_accuracy(loader, model):
    """
    正答率の判定の出力

    :param loader:　データセット
    :param model: モデル
    :return None
    """
    if loader.dataset.train:
        print("Checking accuracy on train data")
    else:
        print("Checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")

    model.train()
