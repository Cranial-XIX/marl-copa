import copy
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from modules.mixer import Mixer
from modules.replay_buffer import ReplayBuffer
from modules.coach import *
from torch.optim import Adam, RMSprop, SGD
from tqdm import tqdm


class QLearner:
    def __init__(self, mac, args):
        self.args = args
        self.method = args.method
        if "aiqmix" in self.method:
            self.imaginary_lambda = args.imaginary_lambda

        self.mac = mac
        self.mixer = Mixer(args)

        # target networks
        self.target_mac = copy.deepcopy(mac)
        self.target_mixer = copy.deepcopy(self.mixer)
        self.disable_gradient(self.target_mac)
        self.disable_gradient(self.target_mixer)
        self.modules = [self.mac, self.mixer,
                        self.target_mac, self.target_mixer]
        self.params = list(self.mac.parameters()) + list(self.mixer.parameters())
        self.optimizer = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.n_params = sum(p.numel() for p in self.mac.parameters() if p.requires_grad) + \
                sum(p.numel() for p in self.mixer.parameters() if p.requires_grad)

        if args.has_coach:
            self.coach = Coach(args)
            self.target_coach = copy.deepcopy(self.coach)
            self.disable_gradient(self.target_coach)

            self.modules.append(self.coach)
            self.modules.append(self.target_coach)
            self.n_params += sum(p.numel() for p in self.coach.parameters() if p.requires_grad)

            coach_params = list(self.coach.parameters())

            if "vi" in self.method:
                self.vi = VI(args)
                self.modules.append(self.vi)
                coach_params += list(self.vi.parameters())

            self.coach_params = coach_params
            self.coach_optimizer = RMSprop(coach_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        print(f"[info] Total number of params: {self.n_params}")
        self.buffer = ReplayBuffer(args.buffer_size)
        self.t = 0

    def disable_gradient(self, module):
        module.eval()
        for p in module.parameters():
            p.requires_grad = False

    def tensorize(self, args):
        o, e, c, m, ms, a, r = args

        device = self.args.device

        o  = torch.Tensor(o).to(device)     # [batch, t, n_agents, observation_dim]
        e  = torch.Tensor(e).to(device)     # [batch, t, n_others, entity_dim]
        c  = torch.Tensor(c).to(device)     # [batch, t, n_agents, attribute_dim]
        m  = torch.Tensor(m).to(device)     # [batch, t, n_agents, n_all]
        ms = torch.Tensor(ms).to(device)    # [batch, t, n_agents, n_all] full observation
        a  = torch.LongTensor(a).to(device) # [batch, t, n_agents]
        r  = torch.Tensor(r).to(device)     # [batch, t,]

        mask = ms.sum(-1, keepdims=True).gt(0).float()

        o  = mask * o
        a  = mask.long().squeeze(-1) * a
        c  = mask * (c - 0.5)
        return o, e, c, m, ms, a, r

    def update(self, logger, step):
        if len(self.buffer) < self.args.batch_size:
            return

        self.t += 1
        o, e, c, m, ms, a, r = self.tensorize(self.buffer.sample(self.args.batch_size))

        T = o.shape[1]-1 # since we have T+1 steps 0, 1, ..., T

        if self.args.has_coach: # get the z_team_t0
            training_team_strategy = self.mac.z_team.clone() # save previous team strategy

            z_t0, mu_t0, logvar_t0 = self.coach(o[:,0], e[:,0], c[:,0], ms[:,0])
            z_t0_target, _, _ = self.target_coach(o[:,0], e[:,0], c[:,0], ms[:,0])
            z_T_target, _, _ = self.target_coach(o[:,T], e[:,T], c[:,T], ms[:,T])

            self.mac.set_team_strategy(z_t0)
            self.target_mac.set_team_strategy(z_t0_target)

        rnn_hidden = self.mac.init_hidden(o.shape[0], o.shape[2]) # [batch, n_agents, dh]

        Q = []
        for t in range(T):
            prev_a = torch.zeros_like(a[:,0]) if t == 0 else a[:,t-1]
            qa, h, h_full, rnn_hidden = self.mac(
                o[:,t], e[:,t], c[:,t], m[:,t], ms[:,t],
                rnn_hidden, prev_a, a[:,t])

            if self.args.has_coach:
                coach_h = self.coach.encode(o[:,t], e[:,t], c[:,t], ms[:,t])
                q = self.mixer.coach_forward(coach_h, qa, ms[:,t])
            else:
                q = self.mixer(o[:,t], e[:,t], c[:,t], qa, ms[:,t])
            Q.append(q.unsqueeze(-1))

        Q  = torch.cat(Q,  -1) # [batch, T]

        with torch.no_grad():
            NQ = []; NQ_ = [];
            rnn_hidden = self.mac.init_hidden(o.shape[0], o.shape[2]) # [batch, n_agents, dh]
            for t in range(T+1):
                if t == T and self.args.has_coach: # update strategy for last step
                    self.target_mac.set_team_strategy(z_T_target)
                prev_a = torch.zeros_like(a[:,0]) if t == 0 else a[:,t-1]

                qa, h, h_full, rnn_hidden = self.target_mac(
                    o[:,t], e[:,t], c[:,t], m[:,t], ms[:,t], rnn_hidden, prev_a)

                qa = qa.max(-1)[0]
                
                if self.args.has_coach:
                    coach_h = self.target_coach.encode(o[:,t], e[:,t], c[:,t], ms[:,t])
                    nq = self.target_mixer.coach_forward(coach_h, qa, ms[:,t])
                else:
                    nq = self.target_mixer(o[:,t], e[:,t], c[:,t], qa, ms[:,t])
                NQ.append(nq.unsqueeze(-1))

            NQ  = torch.cat(NQ,  -1)[:,1:] # [batch, T]
            #if self.args.has_coach:
            #    NQ_ = torch.cat(NQ_, -1)[:,1:] # [batch, T]

        ######################################################################
        # 1a. Bellman error
        ######################################################################
        td_target = r[:,:-1] + self.args.gamma * NQ
        td_error = F.mse_loss(Q, td_target)
        #if self.args.has_coach:
        #    td_error = td_error * 0.5 + \
        #        0.5 * F.mse_loss(Q_, r[:,:-1] + self.args.gamma * NQ_)

        ######################################################################
        # 1b. Imaginary Bellman error
        ######################################################################
        if "aiqmix" in self.method:
            rnn_hidden = self.mac.init_hidden(o.shape[0]*2, o.shape[2])
            im_Q = []

            ma = m[:,0].sum(-1).gt(0).float() # [batch, n_agent, 1]
            me = F.pad(ma, (0, e.shape[2]), "constant", 1)
            mm = torch.bernoulli(torch.rand_like(me))
            mi = ma.unsqueeze(-1).bmm(mm.unsqueeze(1)) + (1-ma).unsqueeze(-1).bmm(1-mm.unsqueeze(1))
            mo = 1 - mi

            for t in range(T):
                prev_a = torch.zeros_like(a[:,0]) if t == 0 else a[:,t-1]
                im_qa, im_h, fmi, fmo, rnn_hidden = self.mac.im_forward(
                    o[:,t], e[:,t], c[:,t], m[:,t], ms[:,t], rnn_hidden, prev_a, mi, mo, a[:,t])

                im_qa = self.mixer.im_forward(o[:,t], e[:,t], c[:,t], fmi, fmo, im_qa, ms[:,t])
                im_Q.append(im_qa.unsqueeze(-1))

            im_Q = torch.cat(im_Q, -1)
            im_td_error = F.mse_loss(im_Q, td_target)
            td_error = (1-self.imaginary_lambda) * td_error + \
                    self.imaginary_lambda * im_td_error

        ######################################################################
        # 2. ELBO
        ######################################################################
        elbo = 0.
        if self.args.has_coach:
            if "vi" in self.method:
                vi_loss = self.vi(o, e, c, m, ms[:,0], a, z_t0)
                p_ = D.normal.Normal(mu_t0, (0.5 * logvar_t0).exp())
                entropy = p_.entropy().clamp_(0, 10).mean()
                elbo += vi_loss * self.args.vi_lambda - entropy * self.args.vi_lambda / 10

        #print(f"td {td_error.item():.4f} l2 {vi2_loss.item():.4f}")
        #print(f"td {td_error.item():.4f} ent {entropy.item():.4f} l2 {vi2_loss.item():.4f}")

        self.optimizer.zero_grad()
        if self.args.has_coach:
            self.coach_optimizer.zero_grad()

        (td_error + elbo).backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        if self.args.has_coach:
            coach_grad_norm = torch.nn.utils.clip_grad_norm_(self.coach_params, self.args.grad_norm_clip)

        self.optimizer.step()
        if self.args.has_coach:
            self.coach_optimizer.step()
            # set back team strategy for rollout
            self.mac.set_team_strategy(training_team_strategy)

        # update target once in a while
        if self.t % self.args.update_target_every == 0:
            self._update_targets()

        if "aiqmix" in self.method:
            logger.add_scalar("im_q_loss", im_td_error.cpu().item(), step)
        if "vi" in self.method:
            logger.add_scalar("vi", vi_loss.item(), step)

        logger.add_scalar("q_loss", td_error.cpu().item(), step)
        try:
            logger.add_scalar("grad_norm", grad_norm.item(), step)
        except:
            logger.add_scalar("grad_norm", grad_norm, step)

    def save_models(self, path):
        torch.save(self.mac.state_dict(), "{}/mac.th".format(path))
        torch.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        torch.save(self.optimizer.state_dict(), "{}/opt.th".format(path))
        if self.args.has_coach:
            torch.save(self.coach.state_dict(), "{}/coach.th".format(path))
            torch.save(self.coach_optimizer.state_dict(), "{}/coach_opt.th".format(path))
        if "vi" in self.method:
            torch.save(self.vi.state_dict(), "{}/vi.th".format(path))

    def load_models(self, path):
        self.mac.load_state_dict(torch.load("{}/mac.th".format(path)))
        self.mixer.load_state_dict(torch.load("{}/mixer.th".format(path)))
        self.optimizer.load_state_dict(torch.load("{}/opt.th".format(path)))
        if self.args.has_coach:
            self.coach.load_state_dict(torch.load("{}/coach.th".format(path)))
            self.coach_optimizer.load_state_dict(torch.load("{}/coach_opt.th".format(path)))
        if "vi" in self.method:
            self.vi.load_state_dict(torch.load("{}/vi.th".format(path)))
        self.target_mac = copy.deepcopy(self.mac)
        self.target_mixer = copy.deepcopy(self.mixer)
        self.disable_gradient(self.target_mac)
        self.disable_gradient(self.target_mixer)
        if self.args.has_coach:
            self.target_coach = copy.deepcopy(self.coach)
            self.disable_gradient(self.target_coach)

    def _update_targets(self):
        self.target_mac.load_state_dict(self.mac.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        if self.args.has_coach:
            self.target_coach.load_state_dict(self.coach.state_dict())
        return

    def cuda(self):
        for m in self.modules:
            m.cuda()

    def cpu(self):
        for m in self.modules:
            m.cpu()
