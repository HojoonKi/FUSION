import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LearnableFourier(nn.Module):
    """
    Learnable 2D Fourier feature transformer for grayscale (CT) images.
    Input: (B, H, W)
    Output: transformed Fourier features (complex)
    Reference: https://github.com/AI4Science-WestlakeU/FourierFlow/blob/main/models/afno2d.py#L118
    """
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01,
                 hard_thresholding_fraction=1.0, hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, \
            f"hidden_size {hidden_size} must be divisible by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        # learnable block weights (real + imag)
        self.w_real = nn.Parameter(self.scale * torch.randn(num_blocks, self.block_size, self.block_size * hidden_size_factor))
        self.w_imag = nn.Parameter(self.scale * torch.randn(num_blocks, self.block_size, self.block_size * hidden_size_factor))
        self.b_real = nn.Parameter(self.scale * torch.randn(num_blocks, self.block_size * hidden_size_factor))
        self.b_imag = nn.Parameter(self.scale * torch.randn(num_blocks, self.block_size * hidden_size_factor))

        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.alpha = 1.0

    def forward(self, x):
        """
        x: (B, H, W)
        returns: transformed Fourier feature (B, H, W//2+1)
        """
        B, H, W = x.shape
        x = x.float()

        # 2D FFT
        x_fft = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')  # (B, H, W//2 + 1)
        orig_freq_dim = x_fft.shape[-1]  # W//2 + 1 = 129

        # Hard thresholding
        H_cut = int(H * self.hard_thresholding_fraction)
        W_cut = int((W // 2 + 1) * self.hard_thresholding_fraction)
        x_fft = x_fft[:, :H_cut, :W_cut]

        # Magnitude-based adaptive weighting (Normalization)
        magnitude = torch.abs(x_fft)
        magnitude_max = magnitude.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        normalized_magnitude = magnitude / magnitude_max
        
        # Padding & reshape
        freq_dim = x_fft.shape[-1]
        total_size = self.num_blocks * self.block_size
        if freq_dim < total_size:
            pad = total_size - freq_dim
            x_fft = F.pad(x_fft, (0, pad))
            normalized_magnitude = F.pad(normalized_magnitude, (0, pad))
            freq_dim = total_size

        # reshape: (B, H_cut, num_blocks, block_size)
        x_fft = x_fft.reshape(B, H_cut, self.num_blocks, self.block_size)
        normalized_magnitude = normalized_magnitude.reshape(B, H_cut, self.num_blocks, self.block_size)
    
        # Adaptive frequency weighting
        adap_freq = torch.pow(normalized_magnitude, self.alpha) * self.gamma

        xr, xi = x_fft.real, x_fft.imag

        # Adaptive blockwise transform
        yr = (1 + adap_freq) * torch.einsum('bhni,nio->bhno', xr, self.w_real) - \
             (1 + adap_freq) * torch.einsum('bhni,nio->bhno', xi, self.w_imag) + self.b_real
        
        yi = (1 + adap_freq) * torch.einsum('bhni,nio->bhno', xi, self.w_real) + \
             (1 + adap_freq) * torch.einsum('bhni,nio->bhno', xr, self.w_imag) + self.b_imag

        # softshrink for sparsity
        yr = F.softshrink(yr, self.sparsity_threshold)
        yi = F.softshrink(yi, self.sparsity_threshold)

        y = torch.complex(yr, yi)
        y = y.reshape(B, H_cut, -1)
    
        # resize to original frequency dimension
        if H_cut < H:
            y = F.pad(y, (0, 0, 0, H - H_cut))
        y = y[:, :, :orig_freq_dim]

        return y
    

class Fourier(nn.Module):
    def __init__(self, lowpass=True):
        super().__init__()
        self.lowpass = lowpass
    
    def extract_features(self, x):
        dtype = x.dtype
        x = x.float()
        B, H, W = x.shape

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        return x
    
    def lowpass_filter(self, x, cutoff=0.5):
        dtype = x.dtype
        x = x.float()
        B, H, W = x.shape

        # 2D FFT (H, W)
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        # x.shape -> (B, H, W//2+1)

        total_modes = H * (W//2 + 1)
        kept_modes = int(total_modes * cutoff)

        # lowpass filtering
        kept_h = int(H * cutoff)
        kept_w = int((W//2 + 1) * cutoff)
        
        x[:, kept_h:, :] = 0  # H
        x[:, :, kept_w:] = 0  # W
        return x
    
    def forward(self, x, cutoff=0.5):
        if self.lowpass:
            return self.lowpass_filter(x, cutoff)
        else:
            return self.extract_features(x)