import collections
import environment
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import torch
import torch.distributions as D
import yaml

from cv2 import VideoWriter, VideoWriter_fourcc

from config.config import Config
from environment.env_wrappers import SubprocVecEnv, DummyVecEnv
from environment.mpe84 import make_env
from modules.agent import Agent
from modules.q_learner import QLearner
from pathlib import Path
from torchvision.utils import save_image
from tqdm import tqdm
from tensorboardX import SummaryWriter

################################################################################
#
# utils functions
#
################################################################################

def make_parallel_env(n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            np.random.seed(seed + rank * 1000)
            env = make_env()
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def prerun(config):
    suffix = f"{config.method}+ctr{config.centralized_every}"
    if "vi" in config.method:
        suffix = suffix + f"+vi{config.vi_lambda}"

    model_dir = Path('./results') / "mpe" / suffix
    run_num = config.seed
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    return run_dir, log_dir

def reset_wrapper(env):
    o = env.reset()
    c = env.get_attributes()
    e = env.get_entities()
    m = env.get_observability()
    split = m.shape[-1] // 2
    mo, ms = m[..., :split], m[...,split:]
    return o, e, c, mo, ms

def step_wrapper(env, actions):
    no, r, d, _ = env.step(actions)
    ne = env.get_entities()
    m = env.get_observability()
    split = m.shape[-1] // 2
    mo, ms = m[..., :split], m[...,split:]
    return no, ne, mo, ms, r.sum(-1), d

def update_config(env, config):
    o = env.reset()
    c = env.get_attributes()
    e = env.get_entities()
    config.observation_dim = o.shape[-1]
    config.attribute_dim = c.shape[-1]
    config.entity_dim = e.shape[-1]
    config.n_actions = 5

################################################################################
#
# human expert trial
#
################################################################################

def approach(env, i, delta):
    agent = env.envs[0].agents[i]
    # normal max speed is divided by 10, consider speeding up and slowing down
    # to reach an object, it equals to use an average of half speed.
    max_v = agent.attribute[-1] / 20
    step = np.array([max_v, max_v])
    step = np.minimum(step, np.abs(delta))
    agent.state.p_pos = agent.state.p_pos + delta / np.sqrt(np.square(delta).sum()) * step

def expert():
    config = Config()
    env = make_parallel_env(1, config.seed)

    all_rewards = []
    for it in tqdm(range(100)):
        o, e, c, m, ms = reset_wrapper(env)
        max_n_agents = o.shape[1]
        n_agents = int((m.sum(-1) > 0).sum())
        n_entities = e.shape[1] - 1
        episode_reward = 0
        for t in range(config.max_steps):
            #frame = env.envs[0].render(mode="rgb_array")[0]
            catch_invade_i = -1
            if ms[0, 0, -1] == 1:
                invader_pos = e[0,-1,:2].reshape(1, 2)
                pos = o[0,:n_agents,2:4]
                delta = np.sqrt(np.square(pos - invader_pos).sum(-1))
                catch_invade_i = np.argmin(delta)
                approach(env, catch_invade_i, invader_pos.reshape(2) - pos[catch_invade_i])
            for i in range(n_agents):
                if i == catch_invade_i:
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
            o, e, m, ms, r, d = step_wrapper(env, actions.reshape(1, -1))
            episode_reward += r.sum()
        all_rewards.append(episode_reward)
    all_rewards = np.array(all_rewards)
    print(f"mean reward {all_rewards.mean()} | std reward {all_rewards.std()}")
    return all_rewards.mean()

################################################################################
#
# train/test functions
#
################################################################################

def render_episodes():
    from PIL import Image
    config = Config()
    n = 1
    env = make_parallel_env(n, 9999)
    update_config(env, config)

    model_path = "./results/mpe/{}/run0" # this is the path to your model

    # setup modules
    mac = Agent(config) # policy
    qlearner = QLearner(mac, config)
    qlearner.load_models(model_path)
    qlearner.cuda()

    all_rewards = []

    for it in range(20):
        save_path = f"imgs/{config.method}/it{it}/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        #fourcc = VideoWriter_fourcc(*'MP4V')
        #video = VideoWriter(f"{save_path}/epi{it+1}.mp4", fourcc, float(12), (700,700))
        o, e, c, m, ms = reset_wrapper(env)
        prev_a = torch.zeros(o.shape[0], o.shape[1]).long().to(config.device)
        rnn_hidden = mac.init_hidden(o.shape[0], o.shape[1])
        prev_z = torch.zeros(o.shape[0], o.shape[1], config.coach_hidden_dim).to(config.device)

        episode_reward = 0
        for t in range(config.max_steps):
            if "full" in config.method:
                m = ms
            o_, e_, c_, m_, ms_ = mac.tensorize(o, e, c, m, ms)
            if config.has_coach and t % config.centralized_every == 0:
               z_team, _, _ = qlearner.coach(o_, e_, c_, ms_)
               mac.set_team_strategy(z_team)

            frame = env.envs[0].render(mode="rgb_array")[0]
            #video.write(np.uint8(frame))
            #if t == 10:
                #print(o[0,:4])
            im = Image.fromarray(frame)
            im.save(f"{save_path}t{t}.jpg")

            actions, rnn_hidden = mac.step(o_, e_, c_, m_, ms_, rnn_hidden, prev_a, epsilon=0.)
            prev_a = torch.LongTensor(actions).to(config.device)
            o, e, m, ms, r, d = step_wrapper(env, actions)
            episode_reward += r.sum()

            #if (t+1) % config.centralized_every == 0 and config.has_coach:
            #    prev_z = z

        all_rewards.append(episode_reward)
        #video.release()
    all_rewards = np.array(all_rewards)
    print(f"mean reward {all_rewards.mean()} | std reward {all_rewards.std()}")
    return all_rewards.mean()

def test_training():
    config = Config()
    n = 1
    env = make_parallel_env(n, 100000)
    update_config(env, config)

    model_path = "./results/mpe/{}/run0" # this is the path to your model

    # setup modules
    mac = Agent(config) # policy
    qlearner = QLearner(mac, config)
    qlearner.load_models(model_path)
    qlearner.cuda()

    all_rewards = []

    #orders = tt_orders = 0
    orders = 0
    tt_orders = 1e-12
    for it in tqdm(range(100)):
        o, e, c, m, ms = reset_wrapper(env)
        rnn_hidden = mac.init_hidden(o.shape[0], o.shape[1])

        episode_reward = 0
        prev_z = None
        for t in range(config.max_steps):
            o_, e_, c_, m_, ms_ = mac.tensorize(o, e, c, m, ms)
            if config.has_coach and t % config.centralized_every == 0:
                _, z_team, logvar = qlearner.coach(o_, e_, c_, ms_)
                if prev_z is None:
                    mac.set_team_strategy(z_team)
                    prev_z = z_team
                else:
                    bs, n = z_team.shape[:2]
                    mask = ms_.sum(-1).gt(0).float()
                    #normal = D.Normal(z_team, (0.5*logvar).exp())
                    #logprob = normal.log_prob(prev_z).sum(-1)
                    #prob = logprob.exp()
                    #broadcast = (prob > 0.001).float()
                    #import pdb; pdb.set_trace()
                    l2 = (z_team - prev_z).pow(2).sum(-1).sqrt()
                    broadcast = (l2 > 5).float()
                    mac.set_part_team_strategy(z_team, broadcast)
                    #import pdb; pdb.set_trace()
                    orders += (broadcast * mask).sum()
                    tt_orders += mask.sum()
                    prev_z = mac.z_team.clone()

            actions, rnn_hidden = mac.step(o_, e_, c_, m_, ms_, rnn_hidden, epsilon=0.)
            o, e, m, ms, r, d = step_wrapper(env, actions)
            episode_reward += r.sum()

        all_rewards.append(episode_reward)
    all_rewards = np.array(all_rewards)
    print(f"broadcast rate {orders/tt_orders}")
    print(f"mean reward {all_rewards.mean()} | std reward {all_rewards.std()}")
    return all_rewards.mean()

def random():
    config = Config()
    n = 8
    env = make_parallel_env(n, 10000)

    all_rewards = []
    for it in tqdm(range(100)):
        episode_reward = 0.
        env.reset()
        for t in range(config.max_steps):
            #frame = env.envs[0].render(mode="rgb_array")[0]
            actions = np.random.randint(5, size=(n, 6))
            _, r, _, _ = env.step(actions)
            #if r.sum() > 0:
            #print(o[0,:,2:4].max(), o[0,:,2:4].min())
            #print(r[0])
            #import pdb; pdb.set_trace()
            episode_reward += r.sum(-1)
        all_rewards.append(episode_reward)
    all_rewards = np.array(all_rewards)
    print(f"mean reward {all_rewards.mean()} | std reward {all_rewards.std()}")
    return all_rewards.mean()

def run():
    config = Config()
    run_dir, log_dir = prerun(config)

    env = make_parallel_env(config.n_rollout_threads, config.seed)
    update_config(env, config)

    config.pprint()

    # setup modules
    mac = Agent(config) # policy
    qlearner = QLearner(mac, config)
    if config.device == "cuda":
        qlearner.cuda()

    train_stats = {
        "reward": [],
    }

    step = 0
    reward_buffer = collections.deque(maxlen=100)

    use_tqdm = True
    n_iters = config.total_steps // config.max_steps // config.n_rollout_threads

    if use_tqdm:
        pbar = tqdm(total=n_iters)

    prev_update_step = 0

    start_epsilon = 1.0
    end_epsilon = 0.05

    delta = -np.log(end_epsilon) / n_iters

    logger = SummaryWriter(log_dir)

    for it in range(n_iters):
        o, e, c, m, ms = reset_wrapper(env)
        prev_a = torch.zeros(o.shape[0], o.shape[1]).long().to(config.device)

        temporal_buffer = collections.deque(maxlen=config.centralized_every+1) # record t=0,1,...T

        episode_reward = 0.
        epsilon = min(start_epsilon, max(end_epsilon, np.exp(-it * delta)))

        rnn_hidden = mac.init_hidden(o.shape[0], o.shape[1])

        for t in range(config.max_steps):
            step += config.n_rollout_threads

            if "full" in config.method:
                m = ms
            if "interval" in config.method and t % config.centralized_every == 0:
                m = ms

            o_, e_, c_, m_, ms_ = mac.tensorize(o, e, c, m, ms)

            if config.has_coach and t % config.centralized_every == 0:
                with torch.no_grad():
                    z_team, _, _ = qlearner.coach(o_, e_, c_, ms_)
                    mac.set_team_strategy(z_team)

            actions, rnn_hidden = mac.step(o_, e_, c_, m_, ms_, rnn_hidden, prev_a, epsilon) # [n_agents,]
            prev_a = torch.LongTensor(actions).to(config.device)

            no, ne, nm, nms, r, d = step_wrapper(env, actions)

            temporal_buffer.append((o, e, c, m, ms, actions, r))
            episode_reward += r

            if t % config.centralized_every == 0 and t > 0:
                O, E, C, M, MS, A, R = map(np.stack, zip(*temporal_buffer))
                for j in range(config.n_rollout_threads):
                    qlearner.buffer.push(O[:,j], E[:,j], C[:,j],
                                         M[:,j], MS[:,j], A[:,j], R[:,j])

            if (step - prev_update_step) >= config.update_every:
                prev_update_step = step
                qlearner.update(logger, step)

            o = no; e = ne; m = nm; ms = nms

        reward_buffer.extend(episode_reward)
        pbar.update(1)
        running_reward_mean = np.array(reward_buffer).mean()
        train_stats["reward"].append((step, running_reward_mean))
        logger.add_scalar("reward", running_reward_mean, step)
        pbar.set_description(f"ep {it:10d} | {running_reward_mean:8.4f} |")

        if (it+1) % 100 == 0 or (it+1 == n_iters):
            with open(f"{log_dir}/stats.npy", 'wb') as f:
                np.save(f, train_stats)
            f.close()
            qlearner.save_models(f"{run_dir}")

    if use_tqdm:
        pbar.close()
    env.close()

if __name__ == "__main__":
    #test()
    run()
    #expert()
    #random()
