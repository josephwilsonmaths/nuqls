#--- Get hyperparameters from config file
import configparser

def get_config(fname, model, dataset):
    config = configparser.ConfigParser()
    config.read(fname)
    field = f'{model}_{dataset}'
    config_dict = {}
    config_dict['n_experiment'] = config.getint(field,'n_experiment')
    config_dict['epochs'] = config.getint(field,'epochs')
    config_dict['lr'] = config.getfloat(field,'lr')
    config_dict['wd'] = config.getfloat(field,'wd')
    config_dict['bs'] = config.getint(field,'bs')
    config_dict['optim'] = config.get(field,'optim')
    config_dict['sched'] = config.get(field,'sched')
    config_dict['S'] = config.getint(field,'S')
    config_dict['nuqls_S'] = config.getint(field,'nuqls_S')
    config_dict['nuqls_epoch'] = config.getint(field,'nuqls_epoch')
    config_dict['nuqls_lr'] = config.getfloat(field,'nuqls_lr')
    config_dict['nuqls_bs'] = config.getint(field,'nuqls_bs')
    config_dict['nuqls_gamma'] = config.getfloat(field,'nuqls_gamma')
    config_dict['cuqls_S'] = config.getint(field,'cuqls_S')
    config_dict['cuqls_epoch'] = config.getint(field,'cuqls_epoch')
    config_dict['cuqls_lr'] = config.getfloat(field,'cuqls_lr')
    config_dict['cuqls_bs'] = config.getint(field,'cuqls_bs')
    config_dict['cuqls_gamma'] = config.getfloat(field,'cuqls_gamma')
    return config_dict

