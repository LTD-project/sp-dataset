import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

class Embedding_layer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.embeddings = nn.Conv2d(self.in_channel, self.out_channel, kernel_size)
        self._init_weights()
        
    def _init_weights(self):
        nn.init.kaiming_normal_(self.embeddings.weight, mode='fan_out', nonlinearity='relu')
        if self.embeddings.bias is not None:
            nn.init.zeros_(self.embeddings.bias)
            
    def forward(self, x_in):
        x = self.embeddings(x_in)
        return x
        

class Learnable_Frequency_Band(nn.Module):
    def __init__(self, dim, num_bands=3, learnable_bands=True, feature_dim=128):
        super().__init__()
        self.dim = dim
        self.num_bands = num_bands
        self.learnable_bands = learnable_bands
        self.feature_dim = feature_dim

        # Learnable or fixed frequency band
        if learnable_bands:
            self.band_boundaries = nn.Parameter(torch.sigmoid(torch.rand(num_bands-1)))
        else:
            self.register_buffer('band_boundaries', torch.linspace(0, 1, num_bands+1)[1:-1])
        
        self.band_weights = nn.Parameter(torch.ones(num_bands))  
        
        self.complex_weights = nn.ParameterList([
            nn.Parameter(torch.randn(dim, feature_dim, 2) * 0.02) 
            for _ in range(num_bands)
        ])
        
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.band_weights, mean=1.0, std=0.2, a=0.5, b=2.0)
        
        for weight in self.complex_weights:
            nn.init.trunc_normal_(weight, std=0.02)

    def get_band_masks(self, num_freq):
        """
        Generate frequency band masking matrix
        return: [num_bands, num_freq]
        """
        device = self.band_boundaries.device
        # 生成频段边界索引
        boundaries = self.band_boundaries if self.learnable_bands else self.band_boundaries
        band_edges = torch.cat([
            torch.tensor([0.0], device=device), 
            boundaries, 
            torch.tensor([1.0], device=device)
        ])
        freq_indices = (band_edges * (num_freq - 1)).long().tolist()
        
        # Generate binary mask
        masks = []
        for i in range(self.num_bands):
            start, end = freq_indices[i], freq_indices[i+1]
            mask = torch.zeros(num_freq, device=device)
            mask[start:end] = 1.0
            masks.append(mask)
        return torch.stack(masks)  
    
    def forward(self, x_in):
        B, C, T, D = x_in.shape
        
        # time-to-fre (RFFT)
        x_fft = torch.fft.rfft(x_in, dim=2, norm='ortho')  # [B, C, F, D]   F = T//2 + 1
        num_freq = x_fft.shape[2]  
        
        # Generate frequency band mask [num_bands, F]
        band_masks = self.get_band_masks(num_freq)  # [num_bands, F]
        
        band_masks = band_masks.view(1, self.num_bands, num_freq, 1)
         
        band_weights = self.band_weights.view(1, self.num_bands, 1, 1)  
        
        # Processing by frequency band
        x_bands = []
        for i in range(self.num_bands):
           
            mask = band_masks[:, i:i+1]  
            
            x_band = x_fft * mask * band_weights[:, i:i+1]  
        
            weight = torch.view_as_complex(self.complex_weights[i])  # [C, D]
            
            x_band = x_band * weight.view(1, C, 1, D)
            
            x_band = x_band.sum(dim=2)  
            
            x_bands.append(x_band)
        
        # Combine each frequency band [B, C, num_bands, D]
        return torch.stack(x_bands, dim=2)


class FreMLP(nn.Module):
    def __init__(self, embed_dim, embed_size, sparsity_threshold=0.01):
        super().__init__()
        self.embed_size = embed_size
        self.sparsity_threshold = sparsity_threshold
        
        self.r = nn.Parameter(torch.randn(embed_dim, embed_size) * 0.02)  
        self.i = nn.Parameter(torch.randn(embed_dim, embed_size) * 0.02) 
        self.rb = nn.Parameter(torch.randn(embed_size) * 0.02)
        self.ib = nn.Parameter(torch.randn(embed_size) * 0.02)
        
        self._init_weights()
        
        
    def _init_weights(self):
        nn.init.kaiming_normal_(self.r, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.i, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.rb)
        nn.init.zeros_(self.ib)

    def forward(self, x_fft):
        B, C, num_freq, D  = x_fft.shape
        
        o1_real = F.relu(
            torch.einsum('bnfc,ce->bnfe', x_fft.real, self.r) - 
            torch.einsum('bnfc,ce->bnfe', x_fft.imag, self.i) + 
            self.rb
        )
        o1_imag = F.relu(
            torch.einsum('bnfc,ce->bnfe', x_fft.imag, self.r) + 
            torch.einsum('bnfc,ce->bnfe', x_fft.real, self.i) + 
            self.ib
        )
        
        y = torch.stack([o1_real, o1_imag], dim=-1)  
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)      
        return y
    
class Short_cut(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv1 = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=kernel_size, padding='same')
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        # CA
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch//8, 1),
            nn.ReLU(),
            nn.Conv2d(out_ch//8, out_ch, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.bn(self.conv1(x))
        x = self.relu(x)
        att_map = self.att(x)
        return x * att_map
    
class Learnable_Frequency_Band_MLP(nn.Module):
    def __init__(self, input_dim, embed_dim, num_bands=3, embed_size=128):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.embed_size = embed_size
        self.fft_length = None
        
        self.spectral_block = Learnable_Frequency_Band(dim=self.input_dim, num_bands=num_bands, feature_dim=self.embed_dim)
        self.fremlp = FreMLP(embed_dim=self.embed_dim, embed_size=self.embed_size)
        

        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        '''
            x : [32, 24, 270, 128]
        '''
        B, C, T, D = x.shape
        self.fft_length = T
        # Step 2
        x_fft = self.spectral_block(x)           
        # Step 3
        x_fremlp = self.fremlp(x_fft)            
        # Step 4
        x_out = torch.fft.irfft(x_fremlp, n=self.fft_length, dim=2, norm='ortho')   
        return x_out
    

class FTFNet_model(nn.Module):
    def __init__(self, sensor_num, input_dim, seq_length, embed_dim, depth, num_bands=3, embed_size=128, hidden_size=256, dropout=0.2, pre_length=96):
        super().__init__()
        self.sensor_num = sensor_num
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.pre_length = pre_length
        self.fft_length = None
        
        self.embed_layer = Embedding_layer(sensor_num, embed_dim)
        self.learnable_frequency_band_mlp = nn.ModuleList([
            Learnable_Frequency_Band_MLP(input_dim=self.input_dim, 
                          embed_dim=self.embed_dim, 
                          num_bands=num_bands,
                          embed_size=self.embed_size)
            for i in range(depth)]
        )
        
        
        self.shortcut = Short_cut(self.embed_dim, self.embed_size, kernel_size=1)
        # Prediction
        
        self.fc = nn.Sequential(
            nn.Linear(self.seq_length * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):

        if len(x.shape) < 4:
            x = x.permute(0, 2, 1).contiguous()
            x = x.unsqueeze(dim=2)
        
        B, C, _, T = x.shape
        # Embedding
        x = x.permute(0, 2, 1, 3).contiguous()
        x = self.embed_layer(x)
        x = x.permute(0, 2, 3, 1).contiguous() 
        base = x    
    
        for learnable_frequency_band_mlp in self.learnable_frequency_band_mlp:
            x = learnable_frequency_band_mlp(x)
        
        x_short = self.shortcut(base.permute(0, 3, 1, 2).contiguous())
        x = x + x_short.permute(0, 2, 3, 1).contiguous()
    
        x = self.fc(x.reshape(B, C, -1)).permute(0, 2, 1)
        return x
    