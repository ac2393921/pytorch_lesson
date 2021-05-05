from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def load_mnist(batch_size, dataset_dir='../dataset/'):
    """
    MNISTデータを呼び出し、loaderとして渡す

    :param batch_size: バッチサイズ
    :param dataset_dir:　datasetのディレクトリの相対パス
    :return:　train_loader, test_loader
    """
    train_dataset = datasets.MNIST(root=dataset_dir, train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = datasets.MNIST(root=dataset_dir, train=False, transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader
