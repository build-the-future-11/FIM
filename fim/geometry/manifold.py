import torch
import torch.nn as nn

class GridManifold:
    """
    Treats grid as continuous manifold
    """

    def __init__(self, H, W, device="cpu"):
        self.H = H
        self.W = W

        xs = torch.linspace(0, 1, W)
        ys = torch.linspace(0, 1, H)

        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        self.coords = torch.stack([grid_x, grid_y], dim=-1).to(device)  # [H, W, 2]

    def pairwise_distances(self):
        """
        returns [HW, HW] distance matrix
        """
        coords = self.coords.view(-1, 2)
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        dist = torch.norm(diff, dim=-1)
        return dist