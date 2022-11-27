import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d, BatchNorm2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from utils import init_weights, get_padding
from stft import TorchSTFT

LRELU_SLOPE = 0.1


class ResBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convs = nn.ModuleList([
            Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            BatchNorm2d(out_ch),
            nn.ReLU()
        ])
        self.out_ch = out_ch
        self.in_ch = in_ch

    def forward(self, x):
        for c in self.convs:
            res = c(x)
            if self.out_ch == self.in_ch:
                x = res + x
            else:
                x = res
        return x
    
class Encoder(torch.nn.Module):
    def __init__(self, h):
        super().__init__()
        
        self.encs = nn.ModuleList()
        middle = h.n_blocks//2 + 1
        for i in range(1, h.n_blocks+1):
            if i < middle:
                self.encs.append(ResBlock(4,4))
            elif i == middle:
                self.encs.append(ResBlock(4,1))
            else:
                self.encs.append(ResBlock(1,1))
                
        self.linear = nn.Linear(h.win_size//2+1,h.latent_dim)
        self.dropout = nn.Dropout(h.latent_dropout)
    
    def forward(self, x):
        # x: (B, 4, N, T)
        for enc_block in self.encs:
            x = enc_block(x)
        
        x = x.squeeze(1).transpose(1,2) # (B, 1, N, T) -> (B, T, N)
        x = self.linear(x)
        #! Apply dropout (according to DAE) to increase decoder robustness,
        #! because representation predicted from AM is used in TTS application.
        x = self.dropout(x)
        
        return x
    
class Generator(torch.nn.Module):
    def __init__(self, h):
        super().__init__()
        
        self.linear = nn.Linear(h.latent_dim, h.win_size//2+1)
        self.decs = nn.ModuleList()
        middle = h.n_blocks//2 + 1
        for i in range(1, h.n_blocks+1):
            if i < middle:
                self.decs.append(ResBlock(1,1))
            elif i == middle:
                self.decs.append(ResBlock(1,4))
            else:
                self.decs.append(ResBlock(4,4))
                
        self.dec_istft_input = h.dec_istft_input
        
        if self.dec_istft_input == 'cartesian' or 'polar':
            self.conv_post = Conv2d(4,2,3,1,padding=1) # Predict Real/Img (default) or Magitude/Phase
        elif self.dec_istft_input == 'both':
            self.conv_post = Conv2d(4,4,3,1,padding=1) # Predict Real/ImgMagitude/Phase
        
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft = TorchSTFT(filter_length=h.n_fft, hop_length=h.hop_size, win_length=h.win_size)
        
    
    def forward(self, x):
        # x: (B, T, D)
        x = self.linear(x)
        x = x.transpose(1,2).unsqueeze(1)
        for dec_block in self.decs:
            x = dec_block(x)
        
        # (B, 4, N, T)
        x = F.leaky_relu(x)
        x = x.contiguous().view(x.size(0),-1,x.size(-1)) # (B, 4N, T)
        x = self.reflection_pad(x)
        x = x.contiguous().view(x.size(0),4,-1,x.size(-1)) # (B, 4N, T') -> (B, 4, N, T')
        # (B, 4, N, T') -> (B, 2, N, T') (default) or (B, 4, N, T')
        x = self.conv_post(x)
        
        if self.dec_istft_input == 'cartesian': #! default
            real = x[:,0,:,:]
            imag = x[:,1,:,:]
            wav = self.stft.cartesian_inverse(real, imag)
        elif self.dec_istft_input == 'polar':
            magnitude = x[:,0,:,:]
            phase = x[:,1,:,:]
            wav = self.stft.polar_inverse(magnitude, phase)
        elif self.dec_istft_input == 'both':
            real = x[:,0,:,:]
            imag = x[:,1,:,:]
            magnitude = x[:,2,:,:]
            phase = x[:,3,:,:]
            wav = self.stft.both_inverse(real, imag, magnitude, phase)
            
        return wav

class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

