import torch
from experiments.exp_forecasting import Exp_Forecast
import random
import numpy as np
from config import args

if __name__ == '__main__':
    # fix seed for reproducibility's sake
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # define training/testing/predicting class
    Exp = Exp_Forecast
    exp = Exp(args)

    # want to train from scratch? 
    args.train_from_scratch = True

    # data path for prediction data - modify as desired
    args.data_path_for_prediction = "./data.csv"

    # training bit
    if args.train_from_scratch:
        print('>>>>>>> Training arguments >>>>>>>')
        for key, value in vars(args).items():
            print(f"{key}: {value}")
        print("\n")
        exp.train()

    # predicting bit
    preds_y = exp.predict(args.train_from_scratch)

    # add seq_len-1 zeros at the front to match the prediction data length
    pad_with_zeros = False
    if pad_with_zeros:
        preds_y = np.concatenate((np.zeros(args.seq_len-1), preds_y))

    np.save("./output/y_predictions.npy", preds_y)

    print('>>>>>>> Finished, results saved in output folder >>>>>>>')

    torch.cuda.empty_cache()
