import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torchvision import datasets
from torch.utils.data import DataLoader

def get_dataset(name, subsample):
    if name=='mnist':
        training_data = datasets.MNIST(
            root="data/MNIST",
            train=True,
            download=True,
            transform=ToTensor()
        )

        test_data = datasets.MNIST(
            root="data/MNIST",
            train=False,
            download=True,
            transform=ToTensor()
        )

        ood_test_data = datasets.FashionMNIST(
            root="data/FashionMNIST",
            train=False,
            download=True,
            transform=ToTensor()
        )

        _, val_data = torch.utils.data.random_split(training_data,[50000,10000])
        n_output = 10
        n_channels = 1

    elif name=='fmnist':
        training_data = datasets.FashionMNIST(
            root="data/FashionMNIST",
            train=True,
            download=True,
            transform=ToTensor()
        )

        test_data = datasets.FashionMNIST(
            root="data/FashionMNIST",
            train=False,
            download=True,
            transform=ToTensor()
        )

        ood_test_data = datasets.MNIST(
            root="data/MNIST",
            train=False,
            download=True,
            transform=ToTensor()
        ) 

        _, val_data = torch.utils.data.random_split(training_data,[50000,10000])
        n_output = 10
        n_channels = 1

    elif name=='cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        training_data = datasets.CIFAR10(
            root="data/CIFAR10",
            train=True,
            download=True,
            transform=transform_train
        )

        test_data = datasets.CIFAR10(
            root="data/CIFAR10",
            train=False,
            download=True,
            transform=transform_test
        )

        ood_test_data = datasets.CIFAR100(
            root="data/CIFAR100",
            train=False,
            download=True,
            transform=transform_test
        ) 

        _, val_data = torch.utils.data.random_split(training_data,[50000,10000])
        n_output = 10
        n_channels = 3

    elif name=='svhn':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
        ])

        training_data = datasets.SVHN(
            root="data/SVHN",
            split='train',
            download=True,
            transform=transform_train
        )

        test_data = datasets.SVHN(
            root="data/SVHN",
            split='test',
            download=True,
            transform=transform_test
        )

        transform_test_cifar = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        ood_test_data = datasets.CIFAR10(
            root="data/CIFAR10",
            train=False,
            download=True,
            transform=transform_test_cifar
        ) 

        _, val_data = torch.utils.data.random_split(training_data,[50000,10000])
        n_output = 10
        n_channels = 3

    elif name=='cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        training_data = datasets.CIFAR100(
            root="data/CIFAR100",
            train=True,
            download=True,
            transform=transform_train
        )

        test_data = datasets.CIFAR100(
            root="data/CIFAR100",
            train=False,
            download=True,
            transform=transform_test
        )

        ood_test_data = datasets.CIFAR10(
            root="data/CIFAR10",
            train=False,
            download=True,
            transform=transform_test
        ) 

        _, val_data = torch.utils.data.random_split(training_data,[50000,10000])
        n_output = 100
        n_channels = 3

    elif name == 'imagenet':
        from torchvision.models import ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT
        preprocess = weights.transforms()
        
        training_data = datasets.ImageFolder(
            root='/scratch/licenseddata/imagenet/imagenet-1k/train',
            transform = preprocess
        )

        mean_test = [0.485, 0.456, 0.406]
        std_test = [0.229, 0.224, 0.225]

        test_transform = transforms.Compose(
            [transforms.Resize(232), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean_test, std_test)])

        test_data = datasets.ImageFolder(
            root='/scratch/licenseddata/imagenet/imagenet-1k/val',
            transform = test_transform
        )

        mean_ood = [0.485, 0.456, 0.406]
        std_ood = [0.229, 0.224, 0.225]

        ood_transform = transforms.Compose(
            [transforms.Resize(232), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean_ood, std_ood)])

        print('loading datasets')

        ood_test_data = datasets.ImageFolder(r'data/imagenet-o/imagenet-o', transform=ood_transform)

        n_output = 1000
        n_channels = 3

        val_data = test_data

        print('loaded datasets')

    if subsample:
        n_train = 1000
        n_test = 100
        training_data = torch.utils.data.Subset(training_data,range(n_train))
        test_data = torch.utils.data.Subset(test_data,range(n_test))
        val_data = torch.utils.data.Subset(val_data,range(n_test))
        ood_test_data = torch.utils.data.Subset(ood_test_data,range(n_test))

    return training_data, test_data, val_data, ood_test_data, n_output, n_channels

class load_dataset(object):
    def __init__(self, name, subsample):
        training_data, test_data, val_data, ood_test_data, n_output, n_channels = get_dataset(name=name, subsample=subsample)
        self.training_data = training_data
        self.test_data = test_data
        self.val_data = val_data
        self.ood_test_data = ood_test_data
        self.n_output = n_output
        self.n_channels = n_channels

    def trainloader(self, batch_size):
        return DataLoader(self.training_data, batch_size=batch_size, shuffle=True)
    
    def testloader(self, batch_size):
        return DataLoader(self.test_data, batch_size=batch_size, shuffle=False)
    
    def valloader(self, batch_size):
        return DataLoader(self.val_data, batch_size=batch_size, shuffle=True)
    
    def oodtestloader(self, batch_size):
        return DataLoader(self.ood_test_data, batch_size=batch_size, shuffle=False)

    

