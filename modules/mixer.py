import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.attention import MHA

class HyperNet(nn.Module):
    def __init__(self, args):
        super(HyperNet, self).__init__()
        self.args = args
        dc = args.attribute_dim
        do = args.observation_dim
        de = args.entity_dim
        dh = args.coach_hidden_dim
        dmh = args.mixer_hidden_dim
        dme = args.mixer_embed_dim

        self.agent_encode = nn.Linear(do+dc, dh)
        self.other_encode = nn.Linear(de, dh)
        self.mha = MHA(input_dim=dh, hidden_dim=dh, n_heads=args.n_heads)
        self.head = nn.Sequential(
            nn.Linear(dh, dmh),
            nn.ReLU(),
            nn.Linear(dmh, dme))

    def forward(self, o, e, c, ms):
        o = torch.cat([o, c], -1)
        ha = self.agent_encode(o) # [batch, n_agents, dh]
        he = self.other_encode(e) # [batch, n_others, dh]
        n_others = he.shape[1]
        x = torch.cat([ha, he], 1) # [batch, n_all, dh]
        hidden = self.mha(x, ms) # [batch, n_agents, dh]
        return self.head(hidden)

class Mixer(nn.Module):
    def __init__(self, args):
        super(Mixer, self).__init__()

        self.args = args
        dh = args.mixer_hidden_dim
        de = args.mixer_embed_dim

        dh_agent = args.agent_hidden_dim
        dh_coach = args.coach_hidden_dim

        if args.has_coach:
            self.coach_mixer_W1 = nn.Sequential(
                nn.Linear(dh_coach, dh),
                nn.ReLU(),
                nn.Linear(dh, de)
            )
            self.coach_mixer_b1 = nn.Sequential(
                nn.Linear(dh_coach, dh),
                nn.ReLU(),
                nn.Linear(dh, de)
            )
            self.coach_mixer_w2 = nn.Sequential(
                nn.Linear(dh_coach, dh),
                nn.ReLU(),
                nn.Linear(dh, de)
            )
            self.coach_mixer_b2 = nn.Sequential(
                nn.Linear(dh_coach, dh),
                nn.ReLU(),
                nn.Linear(dh, de)
            )
        else:
            self.mixer_W1 = HyperNet(args)
            self.mixer_b1 = HyperNet(args)
            self.mixer_w2 = HyperNet(args)
            self.mixer_b2 = HyperNet(args)

    def forward(self, o, e, c, qa, ms):
        """
        h:  [batch, n_agents, strategy_dim] 
        qa: [batch, n_agents]
        ms: [batch, n_agents, n_entities]
        """
        bs, n_agents = qa.shape[:2]
        mask = ms.sum(-1).gt(0).float() # [batch, n_agents]
        qa = qa * mask
        qa = qa.unsqueeze(1) # [batch, 1, n_agents]

        mW1 = self.mixer_W1(o,e,c,ms)
        mb1 = self.mixer_b1(o,e,c,ms)
        mw2 = self.mixer_w2(o,e,c,ms)
        mb2 = self.mixer_b2(o,e,c,ms)

        # all above matrices are of shape [batch, n_agents, de]
        mask = mask.unsqueeze(2) # [batch, n_agents, 1]
        mW1, mw2 = map(lambda p: F.softmax(p, dim=-1), [mW1, mw2])
        mW1 = mW1 * mask
        mw2 = mw2 * mask
        mb1 = mb1 * mask
        mb2 = mb2 * mask

        active_agents = (mask.sum(1) + 1e-12)
        mb1 = mb1.sum(1) / active_agents # [batch, de]
        mw2 = mw2.sum(1) / active_agents # [batch, de]
        mb2 = (mb2.sum(1) / active_agents).mean(-1) # [batch,]

        q = F.elu(qa.bmm(mW1) + mb1.unsqueeze(1)).bmm(mw2.unsqueeze(2)).view(bs) + mb2
        return q # [batch,]

    def coach_forward(self, h, qa, ms):
        """
        h:  [batch, n_agents, strategy_dim] 
        qa: [batch, n_agents]
        m:  [batch, n_agents, n_entities]
        """
        bs, n_agents = qa.shape[:2]
        mask = ms.sum(-1).gt(0).float() # [batch, n_agents]
        qa = qa * mask
        qa = qa.unsqueeze(1) # [batch, 1, n_agents]

        mW1, mb1, mw2, mb2 = self.coach_mixer_W1(h), self.coach_mixer_b1(h), self.coach_mixer_w2(h), self.coach_mixer_b2(h)
        # all above matrices are of shape [batch, n_agents, de]
        mask = mask.unsqueeze(2) # [batch, n_agents, 1]
        mW1, mw2 = map(lambda p: F.softmax(p, dim=-1), [mW1, mw2])
        mW1 = mW1 * mask
        mw2 = mw2 * mask
        mb1 = mb1 * mask
        mb2 = mb2 * mask

        active_agents = (mask.sum(1) + 1e-12)
        mb1 = mb1.sum(1) / active_agents # [batch, de]
        mw2 = mw2.sum(1) / active_agents # [batch, de]
        mb2 = (mb2.sum(1) / active_agents).mean(-1) # [batch,]

        q = F.elu(qa.bmm(mW1) + mb1.unsqueeze(1)).bmm(mw2.unsqueeze(2)).view(bs) + mb2
        return q # [batch,]

    def im_forward(self, o, e, c, fmi, fmo, im_q, ms):
        """
        im_z: [2*batch, n_agents, strategy_dim*2]
        z:    [batch, n_agents, strategy_dim*2]
        im_q: [2*batch, n_agents]
        m:    [batch, n_agents]
        """
        qi, qo = im_q.chunk(2, dim=0)
        bs, n_agents = qi.shape[:2]

        mask = ms.sum(-1).gt(0).float() # [batch, n_agents]
        qi = qi * mask
        qo = qo * mask

        mW1_i = self.mixer_W1(o,e,c,fmi)
        mW1_o = self.mixer_W1(o,e,c,fmo)
        mb1 = self.mixer_b1(o,e,c,ms)
        mw2 = self.mixer_w2(o,e,c,ms)
        mb2 = self.mixer_b2(o,e,c,ms)

        mask = mask.unsqueeze(2) # [batch, n_agents, 1]
        mW1_i, mW1_o, mw2 = map(lambda p: F.softmax(p, dim=-1), [mW1_i, mW1_o, mw2])
        mW1_i = mW1_i * mask
        mW1_o = mW1_o * mask
        mw2   = mw2   * mask
        mb1   = mb1   * mask
        mb2   = mb2   * mask

        active_agents = (mask.sum(1) + 1e-12)
        mb1 = mb1.sum(1) / active_agents # [batch, de]
        mw2 = mw2.sum(1) / active_agents # [batch, de]
        mb2 = (mb2.sum(1) / active_agents).mean(-1) # [batch,]

        qi = qi.unsqueeze(1) # [batch, 1, n_agents]
        qo = qo.unsqueeze(1) # [batch, 1, n_agents]

        qa = torch.cat([qi, qo], 2) # [batch, 1, 2*n_agents]
        mW1 = torch.cat([mW1_i, mW1_o], 1) # [batch, 2*n_agents, de]
        q = F.elu(qa.bmm(mW1) + mb1.unsqueeze(1)).bmm(mw2.unsqueeze(2)).view(bs) + mb2
        return q # [batch,]
