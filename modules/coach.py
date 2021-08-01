import copy
import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from modules.attention import MHA

class Coach(nn.Module):
    def __init__(self, args):
        super(Coach, self).__init__()
        self.args = args
        dc = args.attribute_dim
        do = args.observation_dim
        de = args.entity_dim
        dh = args.coach_hidden_dim

        self.agent_encode = nn.Linear(do+dc, dh)
        self.other_encode = nn.Linear(de, dh)
        self.mha = MHA(input_dim=dh, hidden_dim=dh, n_heads=args.n_heads)

        # policy for continouos team strategy
        self.mean = nn.Linear(dh, dh)
        self.logvar = nn.Linear(dh, dh)

    def encode(self, o, e, c, ms):
        o = torch.cat([o, c], -1)
        ha = self.agent_encode(o) # [batch, n_agents, dh]
        he = self.other_encode(e) # [batch, n_others, dh]
        n_others = he.shape[1]
        x = torch.cat([ha, he], 1) # [batch, n_all, dh]
        hidden = self.mha(x, ms) # [batch, n_agents, dh]
        return hidden

    def strategy(self, h):
        mu, logvar = self.mean(h), self.logvar(h)
        logvar = logvar.clamp_(-10, 0)
        std = (logvar * 0.5).exp()
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def forward(self, o, e, c, ms):
        bs, n_agents = o.shape[:2]
        h = self.encode(o, e, c, ms) # [batch, n_agents, dh]
        z_team, mu, logvar = self.strategy(h)
        return z_team, mu, logvar


###############################################################################
#
# Variational Objectives
#
###############################################################################

class VI(nn.Module):
    # I(z^a ; s^a_t+1:t+T-1 | s_t)
    def __init__(self, args):
        super(VI, self).__init__()
        self.args = args
        dc = args.attribute_dim
        do = args.observation_dim
        dh = args.coach_hidden_dim
        de = args.entity_dim

        self.na = args.n_actions
        self.action_embedding = nn.Embedding(self.na, self.na)
        self.action_embedding.weight.data = torch.eye(self.na).to(args.device)
        for p in self.action_embedding.parameters():
            p.requires_grad = False
        self.agent_encode = nn.Linear(do+dc+self.na, dh)
        self.other_encode = nn.Linear(de, dh)

        self.mha = MHA(input_dim=dh, hidden_dim=dh, n_heads=args.n_heads)

        self.mean = nn.Sequential(
            nn.Linear(dh, dh),
            nn.ReLU(),
            nn.Linear(dh, dh))
        self.logvar = nn.Sequential(
            nn.Linear(dh, dh),
            nn.ReLU(),
            nn.Linear(dh, dh))
        self.dh = dh

    def forward(self, O, E, C, M, ms_0, A, z_t0):
        bs, T, n_agents = O.shape[:3]

        H = []
        z0 = None
        log_prob = 0
        ma = ms_0[:,0].sum(-1).gt(0) # [bs, n_agents, 1]
        z_t0 = z_t0[ma]
        for t in range(T-1):
            o, e, c, m = O[:,t], E[:,t], C[:,t], M[:,t]
            #no, ne
            #prev_a = torch.zeros_like(A[:,0]) if t == 0 else A[:,t-1]
            #prev_a = self.action_embedding(prev_a)
            a = self.action_embedding(A[:,t])
            o = torch.cat([o, c, a], -1)

            if t == 0: m = ms_0
            ha = self.agent_encode(o) # [batch, n_agents, dh]
            he = self.other_encode(e) # [batch, n_others, dh]
            n_others = he.shape[1]

            x = torch.cat([ha, he], 1) # [batch, n_all, dh]
            h = self.mha(x, m) # [batch, n_agents, dh]
            mu, logvar = self.mean(h), self.logvar(h)
            logvar = logvar.clamp_(-10, 0)
            q_t = D.normal.Normal(mu[ma], (0.5 * logvar[ma]).exp())
            log_prob += q_t.log_prob(z_t0).clamp_(-1000, 0).sum(-1)
        return -log_prob.mean()
