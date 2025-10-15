import torch
import tqdm
import numpy as np
import models as model
from torch import nn
from torch.utils.data import DataLoader
from laplace import Laplace
import os
import posteriors.util as pu
import posteriors.swag as swag
import posteriors.nuqls as nuqls
import posteriors.lla_s as lla_s
import posteriors.mc as mc
import utils.metrics as metrics
from posteriors.de import DeepEnsemble
from posteriors.valla.utils.metrics import SoftmaxClassification, OOD, psd_safe_cholesky
from posteriors.valla.src.valla import VaLLAMultiClassBackend
from posteriors.valla.utils.pytorch_learning import fit
import time
import datetime
import argparse
import warnings
from scipy.cluster.vq import kmeans2
from nuqls.posterior import Nuqls
from cuqls.posterior import Cuqls
from posteriors.be.wide_resnet_batchensemble import Wide_ResNet_BatchEnsemble
import posteriors.be.util as be_util
import utils.training
import utils.classification_dataset
import utils.networks
import utils.hyperparameters
import utils.optimizers
import utils.datasets as ds

warnings.filterwarnings('ignore')

torch.set_default_dtype(torch.float64)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"\n Using {device} device")
if device == 'cuda':
    print(f"CUDA version: {torch.version.cuda}")

parser = argparse.ArgumentParser(description='Classification Experiment')
parser.add_argument('--dataset', default='mnist', type=str, help='dataset')
parser.add_argument('--model', default='lenet', type=str, help='model: lenet, resnet9, resnet50')
parser.add_argument('--subsample', action='store_true', help='Use less datapoints for train and test.')
parser.add_argument('--save_var', action='store_true', help='save variances if on (memory consumption)')
parser.add_argument('--pre_load', action='store_true', help='use pre-trained weights (must have saved state_dict() for correct model + dataset)')
parser.add_argument('--nuqls_pre_load', action='store_true', help='use pre-trained weights (must have saved state_dict() for correct model + dataset)')
parser.add_argument('--verbose', action='store_true',help='verbose flag for all methods')
parser.add_argument('--extra_verbose', action='store_true',help='extra verbose flag for some methods')
parser.add_argument('--progress', action='store_false')
args = parser.parse_args()


# Get dataset
dataset = utils.classification_dataset.load_dataset(name=args.dataset, subsample=args.subsample)

#--- Get hyperparameters from config file
config = utils.hyperparameters.get_config('utils/classification.ini', args.model, args.dataset)
# ---

print('Creating dataloader')
train_dataloader = dataset.trainloader(batch_size=config['bs'])
test_dataloader = dataset.testloader(batch_size=config['bs'])
ood_test_dataloader = dataset.oodtestloader(batch_size=config['bs'])
loss_fn = nn.CrossEntropyLoss()

# Setup metrics
methods = ['MAP','NUQLS']
train_methods = ['MAP','BE','NUQLSi','NUQLS','DE']
test_res = {}
train_res = {}
for m in methods:
    if m in train_methods:
        train_res[m] = {'nll': [],
                        'acc': []}
    test_res[m] = {'nll': [],
                  'acc': [],
                  'ece': [],
                  'oodauc': [],
                  'aucroc': [],
                  'time': [],
                  'mem': []}
    
prob_var_dict = {}
prediction_dict = {}
tolerance_acc_dict = {}

# Setup directories
res_dir = f"./results/image_classification/{args.dataset}_{args.model}/"
ct = datetime.datetime.now()
time_str = f"{ct.day}_{ct.month}_{ct.hour}_{ct.minute}"
if not args.subsample:
    res_dir = res_dir + f"_{time_str}/"
else:
    res_dir = res_dir + f"_s_{time_str}/"

if not os.path.exists(res_dir):
    os.makedirs(res_dir)

if args.verbose:
    print(f'Using {args.dataset} dataset, num train points = {len(dataset.training_data)}')

for ei in tqdm.trange(config['n_experiment']):
    print("\n--- experiment {} ---".format(ei))

    if device == 'cuda':
        print('yes')
        torch.cuda.reset_peak_memory_stats()

    for m in methods:
        t1 = time.time()
        if m == 'MAP':
            map_net = utils.networks.get_model(args.model, dataset.n_output, dataset.n_channels).to(device)
            num_weights = sum(p.numel() for p in map_net.parameters() if p.requires_grad)
            if args.verbose:
                print(f"Network parameter count: {num_weights}")

            # Multiple GPUs
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                map_net = nn.DataParallel(map_net)

            if args.pre_load:
                print('Using pre-loaded weights!')
                model_dict = torch.load(f'models/{args.model}_trained_{args.dataset}.pt', weights_only=True, map_location=device)
                if 'params' in model_dict.keys():
                    map_net.load_state_dict(model_dict['params'])
                    map_net.eval()
                    train_res[m]['nll'].append(model_dict['nll'])
                    train_res[m]['acc'].append(model_dict['acc'])
                else:
                    map_net.load_state_dict(model_dict)
                    map_net.eval()
                    train_res[m]['nll'].append(0.0)
                    train_res[m]['acc'].append(1.0)
            else:
                optimizer, scheduler = utils.optimizers.get_optim_sched(map_net, config['optim'], config['sched'], config['lr'], config['wd'], config['epochs'])
                train_loss, train_acc, _, _ = utils.training.training(train_loader=train_dataloader,test_loader=test_dataloader,
                                                                            model=map_net, loss_fn=loss_fn, optimizer=optimizer,
                                                                            scheduler=scheduler,epochs=config['epochs'],
                                                                            verbose=args.verbose, progress_bar=args.progress)
                train_res[m]['nll'].append(train_loss)
                train_res[m]['acc'].append(train_acc)
                model_dict = {'params': map_net.state_dict(),
                              'nll': train_loss,
                              'acc': train_acc}
                torch.save(model_dict, f'models/{args.model}_trained_{args.dataset}.pt')

            if args.dataset == 'imagenet':
                continue
            else:
                id_map_logits = pu.test_sampler(map_net, dataset.test_data, bs=config['bs'], probit=False)
                ood_map_logits = pu.test_sampler(map_net, dataset.ood_test_data, bs=config['bs'], probit=False)

                id_mean = torch.nn.functional.softmax(id_map_logits,dim=1).cpu(); id_var=None
                ood_mean = torch.nn.functional.softmax(ood_map_logits,dim=1).cpu(); ood_var=None


            id_predictions = id_mean; ood_predictions = ood_mean

        elif m == 'NUQLS':
            nuqls_posterior = Nuqls(map_net, task='classification')
            print(f'MEM BEFORE NUQLS: {1e-9*torch.cuda.max_memory_allocated()}')
            network_mean = True
            if not args.nuqls_pre_load:
                loss,acc = nuqls_posterior.train(train=dataset.training_data, 
                                    train_bs=config['nuqls_bs'], 
                                    n_output=dataset.n_output,
                                    S=config['nuqls_S'],
                                    scale=config['nuqls_gamma'], 
                                    lr=config['nuqls_lr'], 
                                    epochs=config['nuqls_epoch'], 
                                    mu=0.9,
                                    threshold=train_res['MAP']['nll'][ei],
                                    verbose=args.verbose,
                                    extra_verbose=args.extra_verbose,
                                    save_weights=f'models/nuqls_{args.model}_{args.dataset}.pt'
                                    )
                # nuqls_posterior.HyperparameterTuning(dataset.val_data, metric = 'varroc-id', left = 1e-2, right = 1e2, its = 100, network_mean = network_mean, verbose=args.extra_verbose)
                train_res[m]['nll'].append(loss)
                train_res[m]['acc'].append(acc)
                pre_load = None
            else:
                train_res[m]['nll'].append(0.0)
                train_res[m]['acc'].append(1.0)
                pre_load = f'models/nuqls_{args.model}_{args.dataset}.pt'
                nuqls_posterior.pre_load(pre_load)
                # nuqls_posterior.HyperparameterTuning(dataset.val_data, metric = 'varroc-id', left = 1e-2, right = 1e2, its = 100, network_mean = network_mean, verbose=args.extra_verbose)

            if args.dataset == 'imagenet':
                res_text = res_dir + f"result.txt"
                results = open(res_text,'w')
                results.write(f"NUQLS: epochs: {config['nuqls_epoch']}; S: {config['nuqls_S']}; lr: {config['nuqls_lr']}; bs: {config['nuqls_bs']}; gamma: {config['nuqls_gamma']}\n")
                results.close()
                import sys; sys.exit(0)

            id_mean, id_var = nuqls_posterior.UncertaintyPrediction(test=dataset.test_data, test_bs=config['nuqls_bs'], network_mean=network_mean)
            ood_mean, ood_var = nuqls_posterior.UncertaintyPrediction(test=dataset.ood_test_data, test_bs=config['nuqls_bs'], network_mean=network_mean)

            var_function = lambda dataset_input : nuqls_posterior.UncertaintyPrediction(test=dataset_input, test_bs=config['nuqls_bs'], network_mean=network_mean)[1] 

        elif m == 'BE':
            beS = 5

            wrn_n = 1
            be_net = Wide_ResNet_BatchEnsemble(channels=dataset.n_channels,
                                               depth=6*wrn_n+4,
                                               widen_factor=1,
                                               dropout_rate=0,
                                               num_classes=dataset.n_output,
                                               num_models=beS,
                                               pool_number=7).to(device)
            be_net.apply(utils.training.init_weights)

            param_core, params_multi = be_util.vectorize_model(net=be_net)
            be_optimizer = torch.optim.Adam([
                {'params': param_core,'weight_decay': config['wd']},
                {'params': params_multi, 'weight_decay': 0.0}
            ], lr=config['lr'])

            train_loss, train_acc = be_util.train(net=be_net,
                          trainloader=train_dataloader,
                          optimizer=be_optimizer,
                          criterion=torch.nn.CrossEntropyLoss(),
                          epochs=int(config['epochs']*1.5),
                          lr=config['lr'],
                          ensemble_size=beS,
                          device=device)
            
            train_res[m]['nll'].append(train_loss)
            train_res[m]['acc'].append(train_acc)
            
            be_net.training = False
            be_net.eval()
            be_net.variance = True

            id_predictions = []
            for x,_ in test_dataloader:
                id_predictions.append(be_net(x))
            id_predictions = torch.cat(id_predictions,dim=1)
            id_mean = id_predictions.softmax(2).mean(0)

            ood_predictions = []
            for x,_ in ood_test_dataloader:
                ood_predictions.append(be_net(x))
            ood_predictions = torch.cat(ood_predictions,dim=1)
            ood_mean = ood_predictions.softmax(2).mean(0)


        elif m == 'DE':
            de_posterior = DeepEnsemble(network=map_net, task='classification', M = 10)
            train_nll, train_acc = de_posterior.train(loader=train_dataloader, 
                                                    lr=config['lr'], 
                                                    wd=config['wd'],
                                                    epochs=config['epochs'], 
                                                    optim_name=config['optim'], 
                                                    sched_name=config['sched'], 
                                                    verbose=args.verbose,
                                                    extra_verbose=args.extra_verbose)
            train_res[m]['nll'].append(train_nll)
            train_res[m]['acc'].append(train_acc)

            id_mean, id_var = de_posterior.UncertaintyPrediction(test_dataloader)
            ood_mean, ood_var = de_posterior.UncertaintyPrediction(ood_test_dataloader)

            var_function = lambda dataset : de_posterior.UncertaintyPrediction(DataLoader(dataset, config['bs']))[1]
    
        # elif m == 'eNUQLS':
        #     S = config['nuqls_S']
        #     id_preds = []
        #     ood_preds = []
        #     for i in range(S):
        #         if S > 10:
        #             nuqls_method = nuqls.classification_parallel(net = model_list[i], train = dataset.training_data, S = S, epochs=config['nuqls_epoch'], lr=config['nuqls_lr'], n_output=dataset.n_output, 
        #                                                     bs=config['nuqls_bs'], bs_test=config['nuqls_bs'], init_scale=config['nuqls_gamma'])
        #             nuqls_predictions, ood_nuqls_predictions, res = nuqls_method.method(dataset.test_data, ood_test = dataset.ood_test_data, mu=0.9, 
        #                                                                                 weight_decay=config['nuqls_wd'], verbose=args.verbose, 
        #                                                                                 progress_bar=args.progress, gradnorm=True) # S x N x C
        #             del nuqls_method
        #         else:
        #             nuqls_predictions, ood_nuqls_predictions, res = nuqls.series_method(net = model_list[i], train_data = dataset.training_data, test_data = dataset.test_data, 
        #                                                                         ood_test_data=dataset.ood_test_data, train_bs = nuqls_bs, test_bs = nuqls_bs, 
        #                                                                         S = S, scale=nuqls_gamma, lr=nuqls_lr, epochs=nuqls_epoch, mu=0.9, 
        #                                                                         wd = nuqls_wd, verbose = False, progress_bar = True) # S x N x C
        #         print(res['loss'])
        #         print(res['acc'])

        #         train_res[m]['nll'].append(res['loss'].detach().cpu().item())
        #         train_res[m]['acc'].append(res['acc'])

        #         id_predictions = nuqls_predictions.softmax(dim=2)
        #         id_preds.append(id_predictions)
        #         ood_predictions = ood_nuqls_predictions.softmax(dim=2)
        #         ood_preds.append(ood_predictions)
        #     id_predictions = torch.cat(id_preds,dim=0)
        #     ood_predictions = torch.cat(ood_preds,dim=0)

        #     id_mean = id_predictions.mean(dim=0)
        #     ood_mean = ood_predictions.mean(dim=0)

        elif m == 'LLA':
            ## LLA definitions
            def predict(dataloader, la, link='probit'):
                py = []
                for x, _ in dataloader:
                    py.append(la(x.to(device), pred_type="glm", link_approx=link))
                return torch.cat(py).cpu()

            la = Laplace(map_net, "classification",
                        subset_of_weights="last_layer",
                        hessian_structure="kron")
            la.fit(train_dataloader)
            la.optimize_prior_precision(
                method="marglik",
                pred_type='glm',
                link_approx='probit',
                progress_bar=args.progress
            )


            T = 1000
            id_mean, id_var, _ = pu.lla_sampler(dataset=dataset.test_data, 
                                                              model = lambda x : la.predictive_samples(x=x,pred_type='glm',n_samples=T), 
                                                              bs = config['bs'])  # id_predictions -> S x N x C

            ood_mean, ood_var, _ = pu.lla_sampler(dataset=dataset.ood_test_data, 
                                                                    model = lambda x : la.predictive_samples(x=x,pred_type='glm',n_samples=T), 
                                                                    bs = config['bs'])  # ood_predictions -> S x N x C
            
            var_function = lambda dataset : pu.lla_sampler(dataset=dataset, 
                                                                    model = lambda x : la.predictive_samples(x=x,pred_type='glm',n_samples=T), 
                                                                    bs = config['bs'])[1]
            
        elif m == 'LLA_S':
            zeta, scale = lla_s.classification(net=map_net, train=dataset.training_data, train_bs=50, k=9, n_output=dataset.n_output, 
                                alpha0=1, alpha_it=5, post_lr=1e-4, post_epochs=100, lin_lr=1e-2, lin_epochs=100, compute_exact=False)
            
            preds = lla_s.classification_prediction(net=map_net, zeta=zeta, test=dataset.test_data, test_bs=50, scale=scale) # S x N x C
            ood_preds = lla_s.classification_prediction(net=map_net, zeta=zeta, test=dataset.ood_test_data, test_bs=50, scale=scale) # S x N x C

            id_mean = preds.softmax(dim=2).mean(dim=0)
            ood_mean = ood_preds.softmax(dim=2).mean(dim=0)
            id_predictions = preds.softmax(dim=2)
            ood_predictions = ood_preds.softmax(dim=2)

        # elif m == 'VaLLA':
        #     num_inducing = 100
        #     seed = 0
        #     prior_std = 1
        #     bb_alpha = 1
        #     iterations = 5000
        #     val_loader = DataLoader(dataset.val_data, batch_size=50)
        #     fixed_prior = False
        #     dtype = torch.float64
        #     generator = torch.Generator(device=device)
        #     generator.manual_seed(2147483647)
        #     lla_probit = True

        #     Z = []
        #     classes = []
        #     for c in range(dataset.n_output):
        #         # s = training_data.inputs[training_data.targets.flatten() == c]
        #         s = train_x[train_y == c]
        #         z = kmeans2(s.reshape(s.shape[0], -1), 
        #                     num_inducing//n_output, minit="points", 
        #                     seed=seed)[0]
        #         z = z.reshape(num_inducing//n_output, 
        #                     *train_x.shape[1:])

        #         Z.append(z)
        #         classes.append(np.ones(num_inducing//n_output) * c)
        #     Z = np.concatenate(Z)
        #     classes = np.concatenate(classes)

        #     from posteriors.valla.src.backpack_interface import BackPackInterface

        #     valla = VaLLAMultiClassBackend(
        #         map_net,
        #         Z,
        #         backend = BackPackInterface(map_net, n_output),
        #         prior_std=prior_std,
        #         num_data=len(training_data),
        #         output_dim=n_output,
        #         track_inducing_locations=True,
        #         inducing_classes=classes,
        #         y_mean=0.0,
        #         y_std=1.0,
        #         alpha = bb_alpha,
        #         device=device,
        #         dtype=dtype,
        #         #seed = args.seed
        #     )

        #     opt = torch.optim.Adam(valla.parameters(recurse=False), lr=1e-3)
        #     valla_train_dataloader = DataLoader(training_data, batch_size=50, shuffle=True)

        #     loss, val_loss = fit(
        #         valla,
        #         valla_train_dataloader,
        #         opt,
        #         val_metrics=SoftmaxClassification,
        #         val_steps=valla.num_data//bs,
        #         val_generator = val_loader,
        #         use_tqdm=args.verbose,
        #         return_loss=True,
        #         iterations=iterations,
        #         device=device,
        #         dtype = dtype
        #     )

        #     # Get MC-sample predictions to get softmax variance
        #     id_preds = []
        #     for x,y in test_dataloader:
        #         _, Fmean, Fvar = valla.test_step(x.to(device), y.to(device))
                
        #         chol = psd_safe_cholesky(Fvar)
        #         z = torch.randn(2048, Fmean.shape[0], Fvar.shape[-1], generator = generator, device = device,
        #                             dtype = dtype)
        #         preds = Fmean + torch.einsum("sna, nab -> snb", z, chol) # S n X n C
        #         id_preds.append(preds.detach())
        #     id_preds = torch.cat(id_preds,dim=1)

        #     ood_preds = []
        #     for x,y in ood_test_dataloader:
        #         _, Fmean, Fvar = valla.test_step(x.to(device), y.to(device))
        #         chol = psd_safe_cholesky(Fvar)
        #         z = torch.randn(2048, Fmean.shape[0], Fvar.shape[-1], generator = generator, device = device,
        #                             dtype = dtype)
        #         preds = Fmean + torch.einsum("sna, nab -> snb", z, chol) # S n X n C
        #         ood_preds.append(preds.detach())
        #     ood_preds = torch.cat(ood_preds,dim=1)

        #     id_mean = id_preds.softmax(dim=2).mean(dim=0)
        #     ood_mean = ood_preds.softmax(dim=2).mean(dim=0)
        #     id_predictions = id_preds.softmax(dim=2)
        #     ood_predictions = ood_preds.softmax(dim=2)

        #     # Using probit approximation may lead to better results
        #     if lla_probit:
        #         id_preds = []
        #         for x,y in test_dataloader:
        #             _, Fmean, Fvar = valla.test_step(x.to(device), y.to(device))
        #             scaled_logits = Fmean/torch.sqrt( 1 + torch.pi/8 * torch.diagonal(Fvar, dim1 = 1, dim2 = 2))
        #             id_preds.append(scaled_logits.detach())
        #         id_preds = torch.cat(id_preds,dim=0)

        #         ood_preds = []
        #         for x,y in ood_test_dataloader:
        #             _, Fmean, Fvar = valla.test_step(x.to(device), y.to(device))
        #             scaled_logits = Fmean/torch.sqrt( 1 + torch.pi/8 * torch.diagonal(Fvar, dim1 = 1, dim2 = 2))
        #             ood_preds.append(scaled_logits.detach())
        #         ood_preds = torch.cat(ood_preds,dim=0)

        #         id_mean = id_preds.softmax(dim=1)
        #         ood_mean = ood_preds.softmax(dim=1)

        elif m == 'SWAG':
            if args.dataset == 'cifar100' or args.dataset == 'svhn':
                swag_lr = config['lr']
                swag_wd = config['wd']
            else:
                swag_lr = config['lr']*1e2
                swag_wd = 0
            swag_net = swag.SWAG(map_net,epochs = config['epochs'],lr = swag_lr, cov_mat = True,
                                max_num_models=config['S'], wd=swag_wd)
            swag_net.train_swag(train_dataloader=train_dataloader,progress_bar=args.verbose)

            T = 100
            id_mean, id_var, _ = pu.swag_sampler(dataset=dataset.test_data,model=swag_net,T=T,n_output=dataset.n_output,bs=config['bs']) # id_predictions -> S x N x C
            ood_mean, ood_var, _ = pu.swag_sampler(dataset=dataset.ood_test_data,model=swag_net,T=T,n_output=dataset.n_output,bs=config['bs']) # ood_predictions -> S x N x C
            
            var_function = lambda input_dataset : pu.swag_sampler(dataset=input_dataset,model=swag_net,T=T,n_output=dataset.n_output,bs=config['bs'])[1]
        

        elif m == 'MC':
            p = 0.1
            T = 10
            dropout_posterior = mc.MCDropout(network=map_net,
                                 p=p)

            id_mean, id_var = dropout_posterior.mean_variance(test_dataloader, samples=T, verbose=args.extra_verbose)
            ood_mean, ood_var = dropout_posterior.mean_variance(ood_test_dataloader, samples=T, verbose=args.extra_verbose)
            
            var_function = lambda dataset : dropout_posterior.mean_variance(DataLoader(dataset, batch_size=config['bs']), samples=T, verbose=args.extra_verbose)[1]

        # Record metrics
        t2 = time.time()
        test_res[m]['time'].append(t2-t1)
        test_res[m]['mem'].append(1e-9*torch.cuda.max_memory_allocated())

        # [ce, acc, ece, oodauc, aucroc, vmsp_dict]
        metrics_m = metrics.compute_metrics(test_loader=test_dataloader, 
                                            id_mean=id_mean, id_var=id_var, 
                                            ood_mean=ood_mean, ood_var=ood_var, 
                                            variance=(m != 'MAP'), sum=False)
        # metrics_m = metrics.compute_metrics(test_dataloader, id_predictions.cpu(), ood_predictions.cpu(), samples=(m != 'MAP'))

        test_res[m]['nll'].append(metrics_m[0])
        test_res[m]['acc'].append(metrics_m[1])
        test_res[m]['ece'].append(metrics_m[2])
        test_res[m]['oodauc'].append(metrics_m[3])
        test_res[m]['aucroc'].append(metrics_m[4])
        if m != 'MAP':
            prob_var_dict[m] = metrics_m[8]
            acc_conf, acc_var = metrics.tolerance_acc(mean=id_mean, variance=id_var, test_loader=test_dataloader)
            tolerance_acc_dict[m] = {'conf': acc_conf,
                                        'var': acc_var}
            if args.dataset == 'mnist' or args.dataset == 'fmnist':
                varroc_rot = metrics.rotated_dataset(id_var, var_function, args.dataset)
            else:
                varroc_rot = 0.00
            test_res[m]['varroc_rot'].append(varroc_rot)

        elif m == 'MAP':
            acc_conf = metrics.tolerance_acc(mean=id_mean, variance=id_var, test_loader=test_dataloader, deterministic=True)
            tolerance_acc_dict[m] = {'conf': acc_conf}

        metrics.print_results(m, test_res, ei, (train_res if m in train_methods else None))

    # Save predictions, variances for plotting
    if args.save_var:
        prob_var_dict = metrics.add_baseline(prob_var_dict,dataset.test_data,dataset.ood_test_data)
        torch.save(prob_var_dict,res_dir + f"prob_var_dict_{ei}.pt")

        metrics.plot_vmsp(prob_dict=prob_var_dict,
                          title=f'{args.dataset} {args.model}',
                          save_fig=res_dir + f"vmsp_plot.pdf")
        
        torch.save(tolerance_acc_dict, res_dir + f"tolerance_acc_dict_{ei}.pt")
        
    metrics.tolerance_plot(tolerance_acc_dict, save_fig=res_dir + f"tolerance_plot.pdf")

## Record results
res_text = res_dir + f"result.txt"
results = open(res_text,'w')

percentage_metrics = ['acc','ece','oodauc','aucroc','varroc']

results.write("Training Details:\n")
results.write(f"MAP/DE: epochs: {config['epochs']}; S: {config['S']}; lr: {config['lr']}; wd: {config['wd']}; bs: {config['bs']}; n_experiment: {config['n_experiment']}\n")
results.write(f"NUQLS: epochs: {config['nuqls_epoch']}; S: {config['nuqls_S']}; lr: {config['nuqls_lr']}; bs: {config['nuqls_bs']}; gamma: {config['nuqls_gamma']}\n")
results.write(f"CUQLS: epochs: {config['cuqls_epoch']}; S: {config['cuqls_S']}; lr: {config['cuqls_lr']}; bs: {config['cuqls_bs']}; gamma: {config['cuqls_gamma']}\n")

if config["n_experiment"] > 1:
    results.write("\nTrain Results:\n")
    for m in train_res.keys():
        results.write(f"{m}: ")
        for k in train_res[m].keys():
            results.write(f"{k}: {np.mean(train_res[m][k]):.4} +- {np.std(train_res[m][k]):.4}; ")
        results.write('\n')
    results.write("\nTest Prediction:\n")
    for m in test_res.keys():
        results.write(f"{m}: ")
        for k in test_res[m].keys():
            if k == 'varroc' and m != 'NUQLS':
                continue
            results.write(f"{k}: {np.mean(test_res[m][k]):.4} +- {np.std(test_res[m][k]):.4}; ")
        results.write('\n')
else:
    results.write("\nTrain Results:\n")
    for m in train_res.keys():
        results.write(f"{m}: ")
        for k in train_res[m].keys():
            results.write(f"{k}: {train_res[m][k][0]:.4}; ")
        results.write('\n')
    results.write("\nTest Prediction:\n")
    for m in test_res.keys():
        results.write(f"{m}: ")
        for k in test_res[m].keys():
            if (k == 'varroc' or k == 'varroc_id' or k == 'varroc_ood' or k == 'varroc_rot') and m == 'MAP':
                continue
            results.write(f"{k}: {test_res[m][k][0]:.4}; ")
        results.write('\n')

results.close()