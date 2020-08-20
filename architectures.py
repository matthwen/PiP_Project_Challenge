# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


class SimpleCNN(torch.nn.Module):
    def __init__(self, n_in_channels: int = 1, n_hidden_layers: int = 1, n_kernels: int = 32, kernel_size: int = 7):
        """Simple CNN with `n_hidden_layers`, `n_kernels`, and `kernel_size` as hyperparameters"""
        super(SimpleCNN, self).__init__()
        
        self.enc = []
        for i in range(n_hidden_layers):
            self.enc.append(torch.nn.Conv2d(in_channels=n_in_channels, out_channels=n_kernels, kernel_size=kernel_size,
                                       bias=True, padding=int(kernel_size/2)))
            self.enc.append(torch.nn.ReLU())
            n_in_channels = n_kernels
        self.enc_hidden_layers = torch.nn.Sequential(*self.enc)

        # self.dec = []
        # for i in range(n_hidden_layers):
        #     self.dec.append(torch.nn.Conv2d(in_channels=n_in_channels, out_channels=n_kernels,
        #                                      kernel_size=kernel_size, bias=True, padding=int(kernel_size / 2)))
        #     self.dec.append(torch.nn.ReLU())
        #     n_in_channels = n_kernels
        # self.dec_hidden_layers = torch.nn.Sequential(*self.dec)

        self.output_layer = torch.nn.Conv2d(in_channels=n_in_channels, out_channels=1,
                                            kernel_size=kernel_size, bias=True, padding=int(kernel_size / 2))


        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
    
    def forward(self, x):
        """Apply CNN to input `x` of shape (N, n_channels, X, Y), where N=n_samples and X, Y are spatial dimensions"""

        #for f in self.cnn:
        #    x = torch.utils.checkpoint.checkpoint(SimpleWrapper(f), x, self.dummy_tensor)
        #cnn_out = torch.utils.checkpoint.checkpoint(self.hidden_layers,x)  # apply hidden layers (N, n_in_channels, X, Y) -> (N, n_kernels, X, Y)
        cnn_out = self.enc_hidden_layers(x)
        pred = self.output_layer(cnn_out)  # apply output layer (N, n_kernels, X, Y) -> (N, 1, X, Y)
        return pred


