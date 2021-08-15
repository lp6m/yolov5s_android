import torch
import torch.nn as nn
import numpy as np

class Detect(nn.Module):
    def __init__(self, nc=80, INPUT_SIZE=640):
        super(Detect, self).__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = 3 #number of outputs per anchor
        self.na = 3
        self.grid = [Detect._make_grid(n, n) for n in [80, 40, 20]]
        self.stride = [8, 16, 32]
        anchorgrid = [
            [10, 13, 16, 30, 33, 23], #80
            [30, 61, 62, 45, 59, 119], #40
            [116, 90, 156, 198, 373, 326] #20
        ]
        self.anchor_grid = [
            torch.from_numpy(np.reshape(np.array(ag, dtype='float32'), (1, 3, 1, 1, 2)))
            for ag in anchorgrid
        ]
    
    def forward(self, x):
        z = []
        for i in range(self.nl):
            # bs, 20, 20, 255
            bs, ny, nx, _ = x[i].shape
            # bs, 20, 20, 3, 85 -> bs, 3, 20, 20, 85
            x[i] = x[i].reshape(bs, ny, nx, 3, 85).permute(0, 3, 1, 2, 4).contiguous()
            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

            y = x[i].sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.view(bs, -1, self.no))
        return torch.cat(z, 1)
    
    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()