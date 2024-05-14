import torch
from types import SimpleNamespace

args = SimpleNamespace(
    # data loader
    data = 'custom',
    data_path = './data.csv',
    # forecasting task
    model_id = 'XTY_1',
    seq_len = 32,
    pred_len = 1,
    d_model = 32,
    n_heads = 16,
    e_layers = 6,
    d_ff = 64,
    learning_rate = 0.0001,
    batch_size = 512,
    dropout = 0.,
    train_epochs = 10,
    # optimization
    activation = 'gelu',
    num_workers = 10,
    loss = 'MSE',
    # GPU
    use_gpu = True,
    gpu = 0,
    use_multi_gpu = False,
    devices = '0,1,2,3',
    # iTransformer
    use_norm = True
)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
