import pandas as pd
from scipy.io import arff
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torchvision import datasets

def read_regression(dataset_str):
    if dataset_str == 'energy':
        df = pd.read_excel('./data/energy/ENB2012_data.xlsx')
    elif dataset_str == 'concrete':
        df = pd.read_excel('./data/concrete/Concrete_Data.xls')
    elif dataset_str == 'kin8nm':
        arff_file = arff.loadarff('./data/kin8mn/dataset_2175_kin8nm.arff')
        df = pd.DataFrame(arff_file[0])
    elif dataset_str == 'naval':
        df = pd.read_csv('./data/navalPropulsion/data.txt', sep=r"   ", header=None)
    elif dataset_str == 'ccpp':
        df = pd.read_excel('./data/ccpp/Folds5x2_pp.xlsx')
    elif dataset_str == 'protein':
        df = pd.read_csv('./data/protein/CASP.csv')
    elif dataset_str == 'wine':
        df = pd.read_csv('./data/wine/winequality-red.csv', sep=r';')
    elif dataset_str == 'yacht':
        df = pd.read_csv('./data/yacht/yacht_hydrodynamics.data', sep=r'\s{1,}')
    elif dataset_str == 'song':
        df = pd.read_csv('./data/song/YearPredictionMSD.txt', sep=',', header=None)
    else:
        print('Invalid dataset str')
    return df


class RegressionDataset(Dataset):
    '''
    Prepare dataset for regression.
    Input the number of features.

    Input:
    - dataset: numpy array

    Returns:
        - Tuple (X,y) - X is a numpy array, y is a double value.
    '''
    def __init__(self, dataset, input_start, input_dim, target_dim, mX=0, sX=1, my=0, sy=1):
        self.X, self.y = dataset[:,input_start:input_dim], dataset[:,target_dim]

        self.X, self.y = (self.X - mX)/sX, (self.y - my)/sy
        self.len_data = self.X.shape[0]

    def __len__(self):
        return self.len_data

    def __getitem__(self, i):
        return self.X[i,:], self.y[i].reshape(-1)
    
class RotationTransform:
    """Rotate the given angle."""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return torchvision.transforms.functional.rotate(x, self.angle)
    
def get_rotated_FMNIST(angle = 0):
    transform_rotate = transforms.Compose([
        transforms.ToTensor(),
        RotationTransform(angle),
    ])

    test_data = datasets.FashionMNIST(
        root="data/FashionMNIST",
        train=False,
        download=True,
        transform=transform_rotate
    )

    return test_data

def get_rotated_MNIST(angle = 0):
    transform_rotate = transforms.Compose([
        transforms.ToTensor(),
        RotationTransform(angle),
    ])

    test_data = datasets.MNIST(
        root="data/MNIST",
        train=False,
        download=True,
        transform=transform_rotate
    )

    return test_data