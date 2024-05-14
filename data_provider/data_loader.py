import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings
from scipy.ndimage import gaussian_filter1d


warnings.filterwarnings('ignore')

class Dataset_TrainTest(Dataset):
    def __init__(self, flag='train', size=None, data_path=None):

        self.seq_len = size[0]
        self.pred_len = size[1]
        assert flag in ['train', 'test']
        type_map = {'train': 0, 'test': 1}
        self.set_type = type_map[flag]

        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path)
        df_raw = df_raw.fillna(0)

        cols = list(df_raw.columns)
        cols.remove("y")
        df_new = df_raw[cols]

        for col in cols:
            if 'askRate' in col:
                bid_col = col.replace('askRate', 'bidRate')
                df_new[f'sum_{col}'] = 0.5 * (df_raw[col] + df_raw[bid_col])
            elif 'askSize' in col:
                bid_col = col.replace('askSize', 'bidSize')
                df_new[f'ratio_{col}'] = (df_raw[bid_col] - df_raw[col]) / (df_raw[bid_col] + df_raw[col])
                df_new[f'ratio2_{col}'] = (df_raw[bid_col]) / (df_raw[bid_col] + df_raw[col])
                df_new[f'ratio3_{col}'] = (df_raw[col]) / (df_raw[bid_col] + df_raw[col])
                df_new[f'diff_{col}'] = (df_raw[bid_col] - df_raw[col])
                df_new[f'sum_{col}'] = (df_raw[bid_col] + df_raw[col])

        df_new.fillna(0, inplace=True)
        df_new.replace([np.inf, -np.inf], 0, inplace=True)
        df_new['modeAskDepth'] =  df_raw.iloc[:,15:30].max(axis=1)
        df_new['modeBidDepth'] =  df_raw.iloc[:,45:60].max(axis=1)
        df_new['modeAskRate'] =  df_raw.iloc[:,0:15].max(axis=1)
        df_new['modeBidRate'] =  df_raw.iloc[:,30:45].max(axis=1)

        df_new['meanAskDepth'] =  df_raw.iloc[:,15:30].mean(axis=1)
        df_new['meanBidDepth'] =  df_raw.iloc[:,45:60].mean(axis=1)
        df_new['meanAskRate'] =  df_raw.iloc[:,0:15].mean(axis=1)
        df_new['meanBidRate'] =  df_raw.iloc[:,30:45].mean(axis=1)

        derivatives = df_new.diff().fillna(0)
        derivatives.columns = [f"{col}_derivative" for col in df_new.columns]
        df_new = pd.concat([df_new, derivatives], axis=1)

        df_new["y"] = df_raw["y"]
        df_raw = df_new

        num_train = int(len(df_raw) * 0.6)
        num_test = int(len(df_raw) * 0.4)
        border1s = [len(df_raw) - num_train - self.seq_len + 1, 0]
        border2s = [-1, num_test]
        # border1s = [0, len(df_raw) - num_test - self.seq_len + 1]
        # border2s = [num_train, -1]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[:]
        df_data = df_raw[cols_data]

        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)


        if self.set_type == 0:
            np.savez("output/data_norm.npz", mean=self.scaler.mean_, scale=self.scaler.scale_)

        self.data_x = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        seq_x = self.data_x[s_begin:s_end,:-1].copy()
        ys = self.data_x[s_end-1,-1]

        return seq_x, ys

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1 #- 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def convolve(self, input):
        output = np.zeros_like(input)
        time_steps = np.arange(32)
        frequency = 4
        sine_wave = np.sin(2 * np.pi * frequency * time_steps / 32)
        for col_idx in range(input.shape[1]):
            convolved_signal = input[:, col_idx] * sine_wave
            output[:, col_idx] = convolved_signal
        return output

    def smooth(self, input):
        output = np.zeros_like(input)
        for col_idx in range(input.shape[1]):
            new_col = gaussian_filter1d(input[:, col_idx], sigma=0.1)
            output[:, col_idx] = new_col
        return output

    def fourier(self, input, modes=1):
        output = np.zeros_like(input)
        for col_idx in range(input.shape[1]):
            fft_col = np.fft.fft(input[:, col_idx])
            fft_col[modes:-modes] = 0
            output[:, col_idx] = np.real(np.fft.ifft(fft_col))
        return output

class Dataset_Predict(Dataset):
    def __init__(self, flag="pred", size=None, data_path=None):

        self.seq_len = size[0]
        self.pred_len = size[1]
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(self.data_path)
        df_raw = df_raw.fillna(0)

        cols = list(df_raw.columns)
        df_new = df_raw[cols]
        for col in df_raw.columns:
            if 'askRate' in col:
                bid_col = col.replace('askRate', 'bidRate')
                df_new[f'sum_{col}'] = 0.5 * (df_raw[col] + df_raw[bid_col])
            elif 'askSize' in col:
                bid_col = col.replace('askSize', 'bidSize')
                df_new[f'ratio_{col}'] = (df_raw[bid_col] - df_raw[col]) / (df_raw[bid_col] + df_raw[col])
                df_new[f'diff_{col}'] = (df_raw[bid_col] - df_raw[col])
                df_new[f'sum_{col}'] = (df_raw[bid_col] + df_raw[col])

        df_new['modeAskDepth'] =  df_raw.iloc[:,15:30].max(axis=1)
        df_new['modeBidDepth'] =  df_raw.iloc[:,45:60].max(axis=1)
        df_new['modeAskRate'] =  df_raw.iloc[:,0:15].max(axis=1)
        df_new['modeBidRate'] =  df_raw.iloc[:,30:45].max(axis=1)

        df_new['meanAskDepth'] =  df_raw.iloc[:,15:30].mean(axis=1)
        df_new['meanBidDepth'] =  df_raw.iloc[:,45:60].mean(axis=1)
        df_new['meanAskRate'] =  df_raw.iloc[:,0:15].mean(axis=1)
        df_new['meanBidRate'] =  df_raw.iloc[:,30:45].mean(axis=1)

        derivatives = df_new.diff().fillna(0)
        derivatives.columns = [f"{col}_derivative" for col in df_new.columns]
        df_new = pd.concat([df_new, derivatives], axis=1)
        df_raw = df_new

        data = df_raw.values

        data_norm = np.load("./output/data_norm.npz")
        self.means = data_norm['mean']
        self.scales = data_norm['scale']
        data = (data - self.means[:-1]) / self.scales[:-1]

        self.data_x = data

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        seq_x = self.data_x[s_begin:s_end,:]

        return seq_x

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform_y(self, data):
        return data * self.scales[-1] + self.means[-1]

    def fourier(self, input, modes=1):
        output = np.zeros_like(input)
        for col_idx in range(input.shape[1]):
            fft_col = np.fft.fft(input[:, col_idx])
            fft_col[modes:-modes] = 0
            output[:, col_idx] = np.real(np.fft.ifft(fft_col))
        return output