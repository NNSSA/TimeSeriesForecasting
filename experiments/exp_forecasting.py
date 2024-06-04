from data_provider.data_factory import data_provider
import torch
import torch.nn as nn
from torch import optim
from model import iTransformer
import os
import time
import warnings
import numpy as np
import wandb

warnings.filterwarnings("ignore")


class Exp_Forecast(object):
    def __init__(self, args):

        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print("Total number of parameters in the model:", total_params, "\n")

        wandb.init(
            # mode="disabled",
            project="XTY",
            name=(
                "model_id: {}, "
                "seq_len: {}, "
                "pred_len: {}, "
                "batch_size: {}, "
                "lr: {}, "
                "d_model: {}, "
                "n_heads: {}, "
                "e_layers: {}, "
                "d_ff: {}, "
                "dropout: {}".format(
                    args.model_id,
                    args.seq_len,
                    args.pred_len,
                    args.batch_size,
                    args.learning_rate,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_ff,
                    args.dropout,
                )
            ),
            # track hyperparameters and run metadata
            config={
                "seq length": "{}".format(args.seq_len),
                "pred length": "{}".format(args.pred_len),
                "d_model": "{}".format(args.d_model),
                "batch_size": "{}".format(args.batch_size),
                "learning_rate": "{}".format(args.learning_rate),
                "n_heads": "{}".format(args.n_heads),
                "e_layers": "{}".format(args.e_layers),
                "d_ff": "{}".format(args.d_ff),
                "dropout": "{}".format(args.dropout),
            },
        )

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device("cuda:{}".format(self.args.gpu))
            print("Use GPU: cuda:{}".format(self.args.gpu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _build_model(self):
        model = iTransformer.Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _rsquared(self, preds, trues):
        return 1.0 - np.mean((trues - preds) ** 2) / np.mean(
            (trues - np.mean(trues)) ** 2
        )

    def train(self):
        print(">>>>>>> Loading training data >>>>>>>")
        train_data, train_loader = self._get_data(flag="train")
        print(">>>>>>> Loading testing data >>>>>>>")
        test_data, test_loader = self._get_data(flag="test")

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        self.model.train()
        for epoch in range(self.args.train_epochs):
            print("\n")
            print(">>>>>>> Start training >>>>>>>")
            iter_count = 0
            train_loss = []
            preds = []
            trues = []
            self.model.train()
            for i, (past_features, current_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                past_features = past_features.float().to(self.device)
                current_y = current_y.float().to(self.device)

                predicted_y = self.model(past_features).squeeze()
                loss1 = criterion(predicted_y, current_y)
                loss1.backward()
                model_optim.step()

                train_loss.append(loss1.item())
                preds.append(predicted_y.detach().cpu().numpy())
                trues.append(current_y.detach().cpu().numpy())

                if (i + 1) % 100 == 0:
                    print(
                        "\tbatch: {0} / {1}, epoch: {2} | loss: {3:.7f}".format(
                            i + 1, len(train_loader), epoch + 1, loss1.item()
                        )
                    )

            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)

            print("Train loss: ", np.mean(train_loss), "\n")
            print("R^2 train data:", self._rsquared(preds, trues))
            print(">>>>>>> Start testing >>>>>>>")
            self.test(test_loader, criterion)
            wandb.log({"train_loss": np.mean(train_loss)})

        torch.save(self.model.state_dict(), "./output/Trained_model.pth")
        wandb.finish()

    def test(self, test_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (past_features, current_y) in enumerate(test_loader):
                past_features = past_features.float().to(self.device)
                current_y = current_y.float().to(self.device)

                predicted_y = self.model(past_features)

                loss = criterion(predicted_y, current_y)
                preds.append(predicted_y.detach().cpu().numpy())
                trues.append(current_y.detach().cpu().numpy())

                total_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\tbatch: {0} / {1}".format(i + 1, len(test_loader)))

        total_loss = np.average(total_loss)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        print("Test loss: ", total_loss)
        print("R^2 test data:", self._rsquared(preds, trues))
        wandb.log({"r-squared": self._rsquared(preds, trues), "test_loss": total_loss})

        self.model.train()
        return total_loss

    def predict(self, train_from_scratch=True):
        print("\n")
        print(">>>>>>> Loading predicting data >>>>>>>")
        predict_data, predict_loader = self._get_data(flag="pred")

        if not train_from_scratch:
            self.model.load_state_dict(torch.load("./output/Trained_model.pth"))

        preds_y = []

        print(">>>>>>> Start predicting >>>>>>>")
        self.model.eval()
        with torch.no_grad():
            for i, past_features in enumerate(predict_loader):
                past_features = past_features.float().to(self.device)

                predicted_y = self.model(past_features)
                preds_y.append(predicted_y.detach().cpu().numpy())

                if (i + 1) % 100 == 0:
                    print("\tbatch: {0} / {1}".format(i + 1, len(predict_loader)))

        preds_y = np.concatenate(preds_y, axis=0)
        preds_y = predict_data.inverse_transform_y(preds_y)

        return preds_y
