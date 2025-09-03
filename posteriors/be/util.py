import torch
import math
import sys
import tqdm

def vectorize_input_target(inputs,targets,ensemble_size,device):
    inputs = torch.cat([inputs for i in range(ensemble_size)], dim=0)
    targets = torch.cat([targets for i in range(ensemble_size)], dim=0)
    return inputs.to(device) , targets.to(device) 

def vectorize_model(net):
    my_list = ['alpha', 'gamma']
    params_multi_tmp= list(filter(lambda kv: (my_list[0] in kv[0]) or (my_list[1] in kv[0]) , net.named_parameters()))
    param_core_tmp = list(filter(lambda kv: (my_list[0] not in kv[0]) and (my_list[1] not in kv[0]), net.named_parameters()))
    params_multi=[param for name, param in params_multi_tmp]
    param_core=[param for name, param in param_core_tmp]
    return param_core, params_multi

def learning_rate(init, epoch, epochs):
    optim_factor = 0
    if(epoch < int(5/6 * epochs)):
        optim_factor = 3
    elif(epoch < int(4/6 * epochs)):
        optim_factor = 2
    elif(epoch < int(2/6 * epochs)):
        optim_factor = 1

    return init*2

def train(net, trainloader, optimizer, criterion, epochs, lr, ensemble_size, device):
    net.train()
    net.training = True
    

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)

    for epoch in tqdm.tqdm(range(epochs)):
        train_loss = 0
        correct = 0
        total = 0

        # Manually adjust learning rate
        # for group in optimizer.param_groups:
        #     group['lr'] = learning_rate(lr, epoch, epochs)

        print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, learning_rate(lr, epoch, epochs)))
        for batch_idx, (inputs, targets) in enumerate(trainloader):

            # Get repeat inputs and targets for each ensemble member
            inputs, targets = vectorize_input_target(inputs=inputs,targets=targets,ensemble_size=ensemble_size,device=device) # GPU settings
            optimizer.zero_grad()
            outputs = net(inputs)               # Forward Propagation
            loss = criterion(outputs, targets)  # Loss
            loss.backward()  # Backward Propagation
            optimizer.step() # Optimizer update
            scheduler.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                    %(epoch, epochs, batch_idx+1,
                        (len(trainloader.dataset)//inputs.shape[0])+1, loss.item(), 100.*correct/total))
            sys.stdout.flush()

    return train_loss / len(trainloader), correct/total


