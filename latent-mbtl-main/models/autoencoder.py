import torch
import torch.nn as nn


class TrajectoryAutoEncoder(nn.Module):
    def __init__(self, feature_dim=61, hidden_size=256, latent_dim=32, seq_len=500):
        super(TrajectoryAutoEncoder, self).__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim

        # Encoder: 2-layer LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )
        self.fc_enc = nn.Linear(hidden_size, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, hidden_size)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )
        self.output_fc = nn.Linear(hidden_size, feature_dim)

    def forward(self, x):
        # Encoder
        enc_out, (h_n, c_n) = self.encoder_lstm(x)
        last_hidden = h_n[-1]
        z = torch.tanh(self.fc_enc(last_hidden))

        # Decoder
        dec_in = self.fc_dec(z)
        dec_in = dec_in.unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, _ = self.decoder_lstm(dec_in)
        recon = self.output_fc(dec_out)

        return recon, z