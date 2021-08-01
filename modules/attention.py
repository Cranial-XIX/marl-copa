import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MHA(nn.Module):
    """
    the class of Multi-Head Attention
    """
    def __init__(self, input_dim, hidden_dim, n_heads):
        super(MHA, self).__init__()
        self.encode = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(input_dim, hidden_dim))
        self.WQs = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim, hidden_dim)) for i in range(n_heads)])
        self.WKs = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim, hidden_dim)) for i in range(n_heads)])
        self.WVs = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim, hidden_dim)) for i in range(n_heads)])
        self.scale = 1. / np.sqrt(hidden_dim)

    def forward(self, x, m):
        """
        x:      [batch, n_entities, input_dim]
        ma:     [batch, n_agents, n_all]
        return: [batch, n_agents, hidden_dim*n_heads]
        """
        n_agents = m.shape[1]

        h = self.encode(x) # [batch, n, hidden_dim]
        ha = h[:,:n_agents].contiguous()

        outputs = []
        for WQ, WK, WV in zip(self.WQs, self.WKs, self.WVs):
            Q = (ha @ WQ.unsqueeze(0)) # [batch, na, hidden_dim]
            K = (h @ WK.unsqueeze(0))  # [batch, n, hidden_dim]
            V = (h @ WV.unsqueeze(0))  # [batch, n, hidden_dim]
            QK_T = Q.bmm(K.transpose(1,2)) * self.scale # [batch, na, n]

            QK_T = F.softmax(QK_T, dim=-1)
            QK_T = QK_T * m
            prob = QK_T / (QK_T.sum(-1, keepdims=True) + 1e-12)
            if torch.isnan(QK_T).sum() > 0:
                import pdb; pdb.set_trace()

            z = prob.bmm(V) # [batch, na, hidden_dim]
            outputs.append(z.unsqueeze(1))
        output = torch.cat(outputs, dim=1) # [batch, n_heads, na, hidden_dim]
        return output.mean(1) # [batch, na, hidden_dim]
