from data_provider.data_loader import Dataset_TrainTest, Dataset_Predict
from torch.utils.data import DataLoader


def data_provider(args, flag):
    Data = Dataset_TrainTest

    if flag == "train":
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        data_path = args.data_path
    elif flag == "test":
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        data_path = args.data_path
    else:
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        data_path = args.data_path_for_prediction
        Data = Dataset_Predict

    data_set = Data(
        data_path=data_path,
        flag=flag,
        size=[args.seq_len, args.pred_len],
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader
