import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

# 1D persistent homology of rips complex
class RipsH1(nn.Module):
    def __init__(self, max_edge_length):
        super(RipsH1, self).__init__()
        self.max_edge_length = max_edge_length

    def forward(self, input):
        rips = gd.RipsComplex(points=input, max_edge_length=self.max_edge_length)
        st = rips.create_simplex_tree(max_dimension=2)
        st.compute_persistence()
        idx = st.flag_persistence_generators()

        # 0-D
        if len(idx[0]) == 0:
            verts = torch.empty((0, 2), dtype=int)
        else:
            verts = torch.tensor(idx[0][:, 1:])
        deaths = torch.norm(input[verts[:,0], :] - input[verts[:,1], :], dim=-1)

        # 1-D
        if len(idx[1]) == 0:
            verts = torch.empty((0, 4), dtype=int)
        else:
            verts = torch.tensor(idx[1][0])
        dgm1 = torch.norm(input[verts[:, (0, 2)]] - input[verts[:, (1, 3)]], dim=-1)

        return deaths, dgm1

# loss based on persistent homology
rips = RipsH1(max_edge_length=0.5)
def loss(pts):
    deaths, dgm1 = rips(pts)
    total_0pers = torch.sum(deaths**2)
    total_1pers = torch.sum(dgm1[:, 1] - dgm1[:, 0])
    # total_0pers = torch.sum(deaths ** 2)
    disk = (pts ** 2).sum(-1) - 1
    disk = torch.max(disk, torch.zeros_like(disk)).sum()
    # return -total_0pers -total_1pers + 1*disk
    return -total_0pers + 1*disk

# sample random points; to be optimized later
pts = (torch.rand((200, 2)) * 2 - 1)
pts.requires_grad = True
plt.figure()
plt.scatter(pts.detach().numpy()[:, 0], pts.detach().numpy()[:, 1])
plt.show()

# optimization
opt = torch.optim.SGD([pts], lr=0.5)
scheduler = LambdaLR(opt,[lambda epoch: 10./(10+epoch)])
for i in range(500):
    opt.zero_grad(), loss(pts).backward(), opt.step(), scheduler.step()
plt.figure()
plt.scatter(pts.detach().numpy()[:, 0], pts.detach().numpy()[:, 1])
plt.show()
