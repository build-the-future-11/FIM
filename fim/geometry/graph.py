import torch
import torch.nn as nn

class GridGraph:
    """
    Converts HxW grid into graph adjacency (4-neighbor or 8-neighbor)
    """

    def __init__(self, H, W, connectivity=4):
        self.H = H
        self.W = W
        self.N = H * W
        self.connectivity = connectivity

        self.edge_index = self._build_edges()

    def _node_id(self, i, j):
        return i * self.W + j

    def _build_edges(self):
        edges = []

        for i in range(self.H):
            for j in range(self.W):
                u = self._node_id(i, j)

                neighbors = [
                    (i-1, j), (i+1, j),
                    (i, j-1), (i, j+1)
                ]

                if self.connectivity == 8:
                    neighbors += [
                        (i-1, j-1), (i-1, j+1),
                        (i+1, j-1), (i+1, j+1)
                    ]

                for ni, nj in neighbors:
                    if 0 <= ni < self.H and 0 <= nj < self.W:
                        v = self._node_id(ni, nj)
                        edges.append((u, v))

        edge_index = torch.tensor(edges).t().long()  # [2, E]
        return edge_index


class GraphPropagation(nn.Module):
    """
    Message passing over graph
    replaces convolution-based diffusion
    """

    def __init__(self, channels):
        super().__init__()
        self.lin = nn.Linear(channels, channels)

    def forward(self, F, edge_index):
        """
        F: [B, C, H, W]
        edge_index: [2, E]
        """

        B, C, H, W = F.shape
        N = H * W

        x = F.view(B, C, N).permute(0, 2, 1)  # [B, N, C]

        src, dst = edge_index
        messages = self.lin(x[:, src])  # [B, E, C]

        out = torch.zeros_like(x)
        out[:, dst] += messages

        return out.permute(0, 2, 1).view(B, C, H, W)