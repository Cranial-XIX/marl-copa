import argparse
import collections
import environment
import gym
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
import time
import torch
import yaml

from cv2 import VideoWriter, VideoWriter_fourcc

from config.config import Config
from environment.env_wrappers import SubprocVecEnv, DummyVecEnv
from environment.mpe84_5 import make_env_5
from environment.mpe84_6 import make_env_6
from environment.mpe84_change import make_env_change

from modules.agent import Agent
from modules.q_learner import QLearner
from pathlib import Path
from torchvision.utils import save_image
from tqdm import tqdm
from main import step_wrapper, reset_wrapper

n_eval = 500

def generate_test_configurations():
    attributes = np.random.rand(1000, 6, 4)
    attributes[...,-1] = attributes[...,-1] * 0.6 + 0.2
    np.save("test.npy", attributes)

def make_parallel_env(n_rollout_threads, seed, fn):
    def get_env_fn(rank):
        def init_env():
            np.random.seed(seed + rank * 1000)
            env = fn()
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def approach(env, i, delta):
    agent = env.envs[0].agents[i]
    # normal max speed is divided by 10, consider speeding up and slowing down
    # to reach an object, it equals to use an average of half speed since the world is small.
    max_v = agent.attribute[-1] / 20
    step = np.array([max_v, max_v])
    step = np.minimum(step, np.abs(delta))
    agent.state.p_pos = agent.state.p_pos + delta / np.sqrt(np.square(delta).sum()) * step

def test_expert(config, fn):
    env = make_parallel_env(1, 9999, fn)

    all_rewards = []
    for n_ep in tqdm(range(n_eval)):
        obs = env.reset() # [batch, n_agents, observation_dim]
        o, e, c, m, ms = reset_wrapper(env)
        max_n_agents = o.shape[1]
        n_agents = int((m.sum(-1) > 0).sum())
        n_entities = e.shape[1] - 1
        episode_reward = 0
        for t in range(145):
            catch_invade_i = -1
            if ms[0, 0, -1] == 1:
                ma = (m.sum(-1) > 0)[0]
                invader_pos = e[0,-1,:2].reshape(1, 2)
                pos = o[0,:,2:4]
                delta = np.sqrt(np.square(pos - invader_pos).sum(-1))
                catch_invade_i = np.argmin(delta * ma + (1-ma)*10)
                approach(env, catch_invade_i, invader_pos.reshape(2) - pos[catch_invade_i])
            for i in range(6):
                if i == catch_invade_i or (not ma[i]):
                    continue
                if o[0,i,5:8].sum() > 0: # hold something, go home
                    approach(env, i, e[0,0,:2] - o[0,i,2:4])
                else:
                    agent_specialty = np.argmax(c[0,i,:3])
                    targets = []
                    for j in range(n_entities):
                        if e[0,j,3+agent_specialty] == 1:
                            targets.append(j)
                    pos = np.array([e[0,j,:2] for j in targets])
                    delta = pos - o[0,i,2:4].reshape(1,2)
                    to = np.square(delta).sum(-1).argmin()
                    approach(env, i, pos[to] - o[0,i,2:4])

            actions = np.zeros((max_n_agents,))
            o, e, m, ms, r, d = step_wrapper(env, actions.reshape(1,-1))
            episode_reward += r.sum()
        all_rewards.append(episode_reward)
    all_rewards = np.array(all_rewards)
    print(f"[EXPERT] mean reward {all_rewards.mean()} | std reward {all_rewards.std()}")
    return all_rewards.mean()

def test_random(config, fn):
    env = make_parallel_env(1, 9999, fn)
    all_rewards = []
    for it in tqdm(range(n_eval)):
        episode_reward = 0.
        env.reset()
        for t in range(145):
            actions = np.random.randint(5, size=(1, 6))
            _, r, _, _ = env.step(actions)
            episode_reward += r.sum(-1)
        all_rewards.append(episode_reward)
    all_rewards = np.array(all_rewards)
    print(f"[RANDOM] mean reward {all_rewards.mean()} | std reward {all_rewards.std()}")
    return all_rewards.mean()

def update_config(env, config):
    o = env.reset()
    c = env.get_attributes()
    e = env.get_entities()
    config.observation_dim = o.shape[-1]
    config.attribute_dim = c.shape[-1]
    config.entity_dim = e.shape[-1]
    config.n_actions = 5

def test_exp(config, fn, exp, threshold=0.):
    env = make_parallel_env(1, 9999, fn)
    update_config(env, config)
    config.method = exp
    k = exp.find("ctr")
    config.centralized_every = int(exp[k+3:k+4])
    config.agent_hidden_dim = 128

    if "coach" in exp:
        config.has_coach = True

    # setup modules
    mac = Agent(config) # policy
    qlearner = QLearner(mac, config)

    R = []
    OR = []

    for run_num in tqdm([0,1,2,3,4]):
        model_path = f"./results/mpe/{exp}/run{run_num}"

        qlearner.load_models(model_path)
        qlearner.cuda()

        reward = 0
        n_orders = 0
        n_total_orders = 1e-12

        for n_ep in range(n_eval):
            o, e, c, m, ms = reset_wrapper(env)
            prev_a = torch.zeros(o.shape[0], o.shape[1]).long().to(config.device)
            rnn_hidden = mac.init_hidden(o.shape[0], o.shape[1])

            prev_z = None

            for t in range(145):
                if "full" in exp:
                    m = ms
                if "interval" in exp and t % config.centralized_every == 0:
                    m = ms
                o_, e_, c_, m_, ms_ = mac.tensorize(o, e, c, m, ms)

                if config.has_coach and t % config.centralized_every == 0:
                    ma = ms_.sum(-1).gt(0).float()
                    with torch.no_grad():
                        _, z_team, _ = qlearner.coach(o_, e_, c_, ms_)
                    if prev_z is None:
                        mac.set_team_strategy(z_team * ma.unsqueeze(-1))
                        prev_z = z_team
                        n_orders += ma.sum().item()
                        n_total_orders += ma.sum().item()
                    else:
                        bs, n = z_team.shape[:2]
                        #normal = D.Normal(z_team, (0.5*logvar).exp())
                        #logprob = normal.log_prob(prev_z).sum(-1)
                        #prob = logprob.exp()
                        #broadcast = (prob > 0.001).float()
                        #import pdb; pdb.set_trace()
                        l2 = (z_team * ma.unsqueeze(-1) - prev_z * ma.unsqueeze(-1)).pow(2).sum(-1).sqrt()
                        broadcast = (l2 > threshold).float()
                        mac.set_part_team_strategy(z_team, broadcast)
                        #import pdb; pdb.set_trace()
                        n_orders += broadcast.sum().item()
                        n_total_orders += ma.sum().item()
                        prev_z = mac.z_team.clone()

                actions, rnn_hidden = mac.step(o_, e_, c_, m_, ms_, rnn_hidden, prev_a, 0)
                prev_a = torch.LongTensor(actions).to(config.device)
                o, e, m, ms, r, d = step_wrapper(env, actions)
                reward += r.sum()

        reward = reward / n_eval
        rate = n_orders / n_total_orders

        R.append(reward)
        OR.append(rate)

    R = np.array(R)
    OR = np.array(OR)
    print(f"{exp:30s}[{threshold:3d}] | muR: {R.mean():.4f} stdR: {R.std()/np.sqrt(5):.4f} | muC: {OR.mean():.4f} stdC: {OR.std()/np.sqrt(5):.4f}")
    return R.mean(), R.std(), OR.mean(), OR.std()

def print_table():
    envs = ["mpe84_5", "mpe84_6", "mpe84_change"]
    for env in envs:
        print("="*80)
        data = np.load(f"/home/liub/Desktop/mount/teamstrategy/test-results/{env}")
        for d in data:
            print(f"{d[0]:30s} | beta={int(d[1]):2d} | muR {float(d[2]):10.2f} | stdR {float(d[3])/np.sqrt(5):5.2f} | muC {float(d[4]):4.2f} | stdC {float(d[5])/np.sqrt(5):4.2f}")
        print("="*80+"\n\n")

if __name__ == "__main__":
    plot_diff()
    config = Config()
    env_fns = {
        "mpe84_5": make_env_5,
        "mpe84_6": make_env_6,
        "mpe84_change": make_env_change,
    }

    fn = env_fns[config.env_name]

    #test_random(config, fn)
    #test_expert(config, fn)

    import time
    t0 = time.time()
    if True:
        experiments = [
            "aqmix+coach+vi+ctr4+l20.001",
        ]

        results = []
        for e in tqdm(experiments):
            if "vi" in e:
                for t in [0, 1, 2, 3, 5, 8, 10]:
                    a,b,c,d = test_exp(config, fn, e, t)
                    results.append((e,t,a,b,c,d))
            else:
                t = 0
                a,b,c,d = test_exp(config, fn, e, t)
                results.append((e,t,a,b,c,d))

        results = np.array(results)
        with open(f"./test-results/{config.env_name}", 'wb') as f:
            np.save(f, results)
        f.close()
    t1=time.time()
    print(t1-t0)
