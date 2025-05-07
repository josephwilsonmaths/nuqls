import torch
import posteriors.nuqlsPosterior.nuqlsRegression
import posteriors.nuqlsPosterior.nuqlsClassification
import posteriors.nuqlsPosterior.nuqlsRegressionFull

def Nuqls(
    network: torch.nn.Module,
    task: str ='regression',
    full_dataset: bool = False
    ):
    
    if task == 'regression':
        if full_dataset:
            return posteriors.nuqlsPosterior.nuqlsRegressionFull.regressionParallelFull(network)
        else:
            return posteriors.nuqlsPosterior.nuqlsRegression.regressionParallel(network)
    
    elif task == 'classification':
        return posteriors.nuqlsPosterior.nuqlsClassification.classificationParallel(network)
    
