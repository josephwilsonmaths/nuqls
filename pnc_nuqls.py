import torch
import nuqls.posterior as nqls
from scipy.stats import t
import torch.nn.functional as F
import numpy as np
import torch.utils.data as data
import torch.optim as optim
from scipy.stats import norm
from importlib import reload
import argparse
import tqdm

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

def generate_data(data_dim, data_length, gt_label = False):
    x0 = torch.rand(data_length, data_dim) * 0.2
    x = torch.sin(x0)
    y = torch.sum(x, -1)
    if not gt_label:
        y += torch.normal(mean=0, std = 0.001, size = y.shape)
    return x0, y

class ImdbData(data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return X, y

    def __len__(self):
        return len(self.y)

class Net(torch.nn.Module):
    def __init__(self, input_channel, hidden_layer):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_channel, hidden_layer)
        self.fc2 = torch.nn.Linear(hidden_layer, 1)
        self.hidden_layer = hidden_layer


    def forward(self, x ):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = x * np.sqrt(2/self.hidden_layer)
        x = self.fc2(x)
        return x

def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight.data, mean=0.0, std=1.0)
        torch.nn.init.zeros_(m.bias)

def train(model, device, train_loader, optimizer, epoch, epochs, verbose):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss_func = torch.nn.MSELoss()
        output = torch.reshape(output, target.shape)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

    if (epoch%500==0 or epoch == epochs) and verbose:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(
            epoch, loss.item()))
        
    return loss.item()

def main():
    parser = argparse.ArgumentParser(description='PyTorch batching and bootstrap')

    parser.add_argument('--num-validation', type=int, default=4, metavar='N',
                        help='number of validation sets used for ensemble')
    parser.add_argument('--repeats', type=int, default=100, metavar='N',
                        help='number of experiment repeated') # 100
    parser.add_argument('--data-dim', type=int, default=16, metavar='N',
                        help='number of validation sets used for ensemble')
    parser.add_argument('--data-length', type=int, default=1024, metavar='N',
                        help='number of validation sets used for ensemble')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=80, metavar='N',
                        help='number of epochs to train') # 50000
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', # 0.01, 100
                        help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-10, metavar='WD',
                        help='weight decay')
    parser.add_argument('--cl', type=float, default=0.05, metavar='CL',
                        help='confidence level')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    
    parser.add_argument('--width-mult', type=int, default=1, metavar='wM',
                        help='increase width of network')
    parser.add_argument('--nuqls-mult', type=float, default=1, metavar='NM',
                        help='increase width of nuqls variance')
    parser.add_argument('--verbose', action='store_true', default=False)

    args = parser.parse_args()

    print(f'\ndim: {args.data_dim}, n: {args.data_length}')
    cl = 0.05
    ci = 0.95
    ci2 = 0.9

    T_POINT = torch.ones(1,args.data_dim) * 0.1
    T_y = torch.sin(T_POINT)
    T_y = torch.sum(T_y, -1)
    T_POINT = T_POINT

    N_boot = 4
    q1 = norm.ppf((1+ci)/2)
    q2 = norm.ppf((1+ci2)/2)
    c_boot1, c_boot2 = 0, 0
    width_boot1, width_boot2 = 0, 0
    pred_boot = 0
    std_bootstrap_list = []
    mean_bootstrap_list = []

    for repeat in tqdm.tqdm(range(args.repeats)):
        # generate the dataset
        # split the dataset into two part, A and B.
        X_train, y_train = generate_data(data_dim = args.data_dim, data_length = args.data_length)

        # generate dataset A
        A_start_index, A_end_index = 0, args.data_length
        A_set_index = list(range(A_start_index, A_end_index))
        X_Aset, Y_Aset = X_train[A_set_index], y_train[A_set_index]

        # get the initial network
        model0 = Net(input_channel = X_train.shape[-1], hidden_layer=args.data_length * args.width_mult).to(device)
        model0.apply(weight_init)
        optimizer = optim.SGD(model0.parameters(), lr=args.lr, weight_decay=args.wd)
        PATH_ini = 'model0_initial.pth'
        torch.save(model0.state_dict(),PATH_ini)

        # train base models
        l = torch.nn.MSELoss()
        if args.verbose:
            print('Training the base model')
        A_data_base = ImdbData(X_Aset, Y_Aset)
        A_loader_base = torch.utils.data.DataLoader(dataset=A_data_base,
                                                    batch_size=args.batch_size,
                                                    shuffle=True)
        for epoch in range(1, args.epochs + 1):
            tl = train(model0, device, A_loader_base, optimizer, epoch, args.epochs, args.verbose)
        if args.verbose:
            print(f'train_error: {tl:.4}, test error: {l(model0(T_POINT.to(device)),T_y.to(device)).item():.4}')
        if args.save_model:
            PATH_final = 'model0_final.pth'
            torch.save(model0.state_dict(), PATH_final)

        # Create validation set
        X_val, y_val = generate_data(data_dim = args.data_dim, data_length = min(256, args.data_length))
        A_start_index, A_end_index = 0, args.data_length
        A_set_index = list(range(A_start_index, A_end_index))
        X_Aset, Y_Aset = X_val[A_set_index], y_val[A_set_index]
        val_set = ImdbData(X_Aset, Y_Aset)

        nuqls_posterior = nqls.Nuqls(model0, task='regression', full_dataset=True)
        # for epochs in torch.arange(50,400,50):
        #     res = nuqls_posterior.train(train=A_data_base, 
        #                         scale=1e-2, 
        #                         S=10, 
        #                         epochs=int(epochs.item()), 
        #                         lr=1, 
        #                         mu=0.9, 
        #                         verbose=True)
        #     print(f'test error: {l(nuqls_posterior.eval(T_POINT).mean(),T_y).item()}')
        # nuqls_posterior.HyperparameterTuning(validation=val_set, left=0.01, right=100, its=100, verbose=False)

        res = nuqls_posterior.train(train=A_data_base, 
                                scale=1e-2, 
                                S=10, 
                                epochs=200, 
                                lr=1, 
                                mu=0.9, 
                                verbose=args.verbose)
        
        nuqls_posterior.HyperparameterTuning(validation=val_set, left=0.01, right=100, its=200, verbose=False)
        
        pred = nuqls_posterior.eval(T_POINT).cpu()
        scaled_pred = pred * (nuqls_posterior.scale_cal * args.nuqls_mult)
        mean_boot, std_boot = pred.mean(), scaled_pred.std()
        std_bootstrap_list.append(std_boot)
        mean_bootstrap_list.append(mean_boot)

        width_boot1 += q1 * std_boot *2
        width_boot2 += q2 * std_boot *2
        pred_boot += mean_boot
        if T_y >= mean_boot - std_boot * q1 and T_y <= mean_boot + std_boot * q1:
            c_boot1 += 1
        if T_y >= mean_boot - std_boot * q2 and T_y <= mean_boot + std_boot * q2:
            c_boot2 += 1

        # print(f'T_y: {T_y.item():.3}, mean: {mean_boot.item():.3}, MAP mean: {model0(T_POINT).item():.3}, ci: ({(mean_boot - std_boot * q1):.3},{(mean_boot + std_boot * q1):.3})')
        if args.verbose:
            print(f'T_y: {T_y.item():.3}, nuql mean: {mean_boot.item():.3}, model mean: {model0(T_POINT.to(device)).item():.3}, width: {(2* std_boot * q1):.3}')

        if args.verbose:
            print('repeats {}, 95% c_values: {:.2f}, 90% c_values: {:.2f}, width: {:.4f}, pred: {:.4f}' 
            .format(repeat, c_boot1/len(std_bootstrap_list), c_boot2/len(std_bootstrap_list), width_boot2/len(std_bootstrap_list), pred_boot/len(std_bootstrap_list)))
   
    print('95% c_values: {}, width: {:.4f}, 90% c_values: {} width: {:.4f}, pred: {:.4f}' 
        .format(c_boot1/args.repeats, width_boot1/args.repeats, 
        c_boot2/args.repeats, width_boot2/args.repeats, pred_boot/args.repeats))

if __name__ == '__main__':
    main()