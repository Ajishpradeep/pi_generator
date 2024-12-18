import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class RealNVPCouplingLayer(nn.Module):
    def __init__(self, d, mask, hidden_dim=256):
        super().__init__()
        self.d = d
        self.mask = mask
        self.scale_net = MLP(d, hidden_dim, d)
        self.translate_net = MLP(d, hidden_dim, d)

    def forward(self, x, reverse=False):
        self.mask = self.mask.to(x.device)
        x1 = x * self.mask
        x2 = x * (1 - self.mask)

        s = self.scale_net(x1) * (1 - self.mask) 
        t = self.translate_net(x1) * (1 - self.mask)

        if not reverse:
            y2 = x2 * torch.exp(s) + t
            y = x1 + y2
            log_det_jacob = torch.sum(s, dim=1)
        else:
            y2 = x2
            x2 = (y2 - t) * torch.exp(-s)
            y = x1 + x2
            log_det_jacob = -torch.sum(s, dim=1)

        return y, log_det_jacob

class RealNVP(nn.Module):
    def __init__(self, dim=5, num_layers=6, hidden_dim=256):
        super().__init__()
        masks = []
        for i in range(num_layers):
            mask = torch.zeros(dim)
            if i % 2 == 0:
                mask[:dim // 2] = 1
            else:
                mask[dim // 2:] = 1
            masks.append(mask.view(1, -1))

        self.layers = nn.ModuleList([
            RealNVPCouplingLayer(dim, mask=masks[i], hidden_dim=hidden_dim) for i in range(num_layers)
        ])

        self.register_buffer('base_mean', torch.zeros(dim))
        self.register_buffer('base_log_std', torch.zeros(dim))

    def forward(self, x):
        log_det_total = torch.zeros(x.size(0), device=x.device)
        y = x
        for layer in self.layers:
            y, log_det = layer(y, reverse=False)
            log_det_total += log_det
        z = y
        return z, log_det_total

    def inverse(self, z):
        y = z
        for layer in reversed(self.layers):
            y, _ = layer(y, reverse=True)
        x = y
        return x

    def log_prob(self, x):
        z, log_det = self.forward(x)
        log_pz = -0.5 * torch.sum(z ** 2, dim=1) - 0.5 * z.size(1) * torch.log(torch.tensor(2 * torch.pi))
        return log_pz + log_det

    def sample(self, n):
        z = torch.randn(n, self.base_mean.shape[0], device=self.base_mean.device)
        x = self.inverse(z)
        return x
