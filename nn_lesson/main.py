import torch

from model.nn import NN
from model.cnn import CNN
from dataset import load_mnist
from train import train, check_accuracy

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# パラメータの設定
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1


def main():
    train_loader, test_loader = load_mnist(batch_size=batch_size)
    nn = train(NN, input_size, num_classes, learning_rate, num_epochs, train_loader)
    cnn = train(CNN, input_size, num_classes, learning_rate, num_epochs, train_loader)
    check_accuracy(train_loader, nn)
    check_accuracy(test_loader, nn)
    check_accuracy(train_loader, cnn)
    check_accuracy(test_loader, cnn)


if __name__ == '__main__':
    main()
