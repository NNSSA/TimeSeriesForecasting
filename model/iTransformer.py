import torch
import torch.nn as nn
from model.Transformer_EncDec import Encoder, EncoderLayer
from model.SelfAttention_Family import FullAttention, AttentionLayer
from model.Embed import DataEmbedding_inverted


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.dropout
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.projector2 = nn.Linear(316, 50, bias=True)
        self.projector3 = nn.Linear(50, 1, bias=True)

    def forecast(self, x_enc):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variates (tokens)

        # Embedding
        # B L N -> B N E
        enc_out = self.enc_embedding(x_enc)

        # B N E -> B N E
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )
            dec_out = dec_out + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )

        # B S N -> B S 1
        dec_out = self.projector2(dec_out)
        dec_out = self.relu(dec_out)
        dec_out = self.dropout(dec_out)
        dec_out = self.projector3(dec_out)

        return dec_out

    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)
        return dec_out.squeeze()  # squeeze(1).squeeze(1)
