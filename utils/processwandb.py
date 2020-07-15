import wandb

def processWandb(params):
    wandb.init(name=params['run_name'], project="Cityscapes Segmentation")
    # wandb.watch(model, log='all')
    config = wandb.config

    config.batch_size = params['batch_size']
    config.epochs = params['n_epochs']
    config.W = params['W']
    config.H = params['H']
    config.data_root = params['data_root']
    config.train_annot = params['train_annot']
    config.val_annot = params['val_annot']
    config.ckpt_dir = params['ckpt_dir']
    config.log_interval = 1
    config.optimizer = params['opt_name']
    config.lr = params['lr']
    config.b1 = params['b1']
    config.b2 = params['b2']
    config.eps = params['eps']
    config.amsgrad = params['amsgrad']
