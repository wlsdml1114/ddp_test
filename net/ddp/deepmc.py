import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import dropout, nn

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)

class LSTMstack(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.sequence= nn.Sequential(
            nn.Conv1d(1,1,4,padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(),
            nn.Conv1d(1,1,4,padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(),
            nn.LSTM(input_size = 1, hidden_size = 1, dropout = 0.2),
            nn.ReLU()
        )
    def forward(self, WPD):
        return self.sequence(WPD)

class CNNstack(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.sequence= nn.Sequential(
            nn.Conv1d(1,1,4,padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(),
            nn.Conv1d(1,1,4,padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(),
            nn.Conv1d(1,1,4,padding='same'),
            nn.ReLU(),
            nn.Flatten()
        )
    def forward(self, WPD):
        return self.sequence(WPD)

class DeepMC(pl.LightningModule):

    def __init__(self, seq_len = 24, lr = 1e-3):
        super().__init__()
        self.lr = lr
        self.loss = F.mse_loss
        self.seq_len = seq_len
        self.longscale = LSTMstack(self.seq_len)
        self.medium_1 = CNNstack(self.seq_len)
        self.medium_2 = CNNstack(self.seq_len)
        self.medium_3 = CNNstack(self.seq_len)
        self.shotscale = CNNstack(self.seq_len)
        self.attention_1 = Attention(self.seq_len,self.seq_len)
        self.attention_2 = Attention(self.seq_len,self.seq_len)

    def forward(self, batch):
        s1,m1,m2,m3,l1,y = batch
        h_j = self.longscale(l1)
        om1 = self.medium_1(m1)
        om2 = self.medium_1(m2)
        om3 = self.medium_1(m3)
        h_short = self.medium_1(s1)


        return [embedding,output]

    def training_step(self, batch, batch_idx):
        x,y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self.loss(x_hat, x)
        self.log("training_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self.loss(x_hat, x)
        self.log("validation_loss", loss, on_step=True, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x,y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self.loss(x_hat, x)
        self.log("test_loss", loss, on_step=True, on_epoch=True, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch
        z = self.encoder(x)
        return self.decoder(z)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer