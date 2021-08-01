import copy
import numpy as np
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario
from multiagent.environment import MultiAgentEnv

class Scenario(BaseScenario):
    def make_world(self):
        world = World()

        # set any world properties first
        world.dim_c = 2

        # add agents
        num_agents = 6
        world.agent_attributes = []
        world.other_entities = []
        world.observability_mask = np.zeros((6, 28))
        world.observable_range = 0.2 # each agent can see this far

        world.agents = [Agent() for i in range(num_agents)]

        for i, agent in enumerate(world.agents):
            agent.i = i
            agent.alive = True
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.at = -1
            agent.hold = 0
            agent.movable = True
            agent.state.p_pos = np.zeros((2))
            agent.state.p_vel = np.zeros((2))
            agent.attribute = np.zeros((5))
            agent.size = 0.04

        # add other objects as landmarks [home, prey1, prey2, prey3, market]
        num_landmarks = 8
        world.landmarks = [Landmark() for i in range(num_landmarks)]

        for i, landmark in enumerate(world.landmarks):
            if i == 0:
                landmark.i = i + num_agents
                landmark.name = 'home'
                landmark.alive = True
                landmark.collide = True
                landmark.movable = False
                landmark.size = 0.08
                landmark.load = 2
                landmark.boundary = False
                landmark.state.p_pos = np.array([0, 0])
                landmark.color = np.array([0,0,0])
            elif i < 7:
                landmark.i = i + num_agents
                landmark.name = str(i)
                landmark.alive = True
                landmark.collide = True
                landmark.movable = False
                landmark.size = 0.05
                landmark.boundary = False
                landmark.color = np.array([0,0,1])
                landmark.state.p_pos = self.get_new_respawn_position()
            elif i == 7:
                landmark.i = i + num_agents
                landmark.name = 'invader'
                landmark.alive = True
                landmark.collide = True 
                landmark.movable = False
                landmark.size = 0.03
                landmark.color = np.array([0., 0., 0.])
                landmark.state.p_pos = np.array([0,0])
                landmark.boundary = False

        self.invader_respawn_time = 0
        self.n_entities = num_agents + num_landmarks

        # add walls
        world.walls = [
            Wall('H', 1.4, width=1),
            Wall('H', -1.4, width=1),
            Wall('V', 1.4, width=1),
            Wall('V', -1.4, width=1),
        ]

        all_choices = []
        for p1 in [0.1, 0.5, 0.9]:
            for p2 in [0.1, 0.5, 0.9]:
                for p3 in [0.1, 0.5, 0.9]:
                    for s in [0.3, 0.5, 0.7]:
                        all_choices.append([p1, p2, p3, s])

        self.all_choices = np.array(all_choices)
        self.num_agents = 4

        return world

    def reset_world(self, world):

        self.num_agents += 1
        if self.num_agents > 4:
            self.num_agents = 2

        # set the initial positions
        num_agents = self.num_agents
        th = np.arange(num_agents) / num_agents * np.pi * 2
        x, y = np.cos(th).reshape(-1, 1), np.sin(th).reshape(-1, 1)
        x = x * 0.12
        y = y * 0.12
        positions = np.concatenate([x, y], -1) # [num_agents, 2]
        world.agent_attributes = []

        max_agents = 6
        for i, agent in enumerate(world.agents):
            agent.i = i
            agent.state.p_pos = np.zeros((2,))
            agent.state.p_vel = np.zeros((2,))
            agent.state.c = np.zeros(world.dim_c)
            flag = (i < num_agents)
            agent.alive = flag
            agent.at = -1
            agent.movable = flag

            attribute = self.all_choices[np.random.randint(self.all_choices.shape[0])]

            #attribute = np.zeros((4,))
            #attribute[i%3] = 1
            #attribute[-1] = 0.5

            agent.attribute = np.array([attribute[0], attribute[1], attribute[2], attribute[3]])

            world.agent_attributes.append(agent.attribute)

            if not agent.alive:
                agent.color = np.zeros((3,))
                continue

            agent.state.p_pos = positions[i]
            agent.accel = attribute[-1] * 5
            agent.max_speed = attribute[-1]

            agent.hold = 0
            agent.color = np.array([attribute[0], attribute[1], attribute[2]])

        world.other_entities = []
        for i, landmark in enumerate(world.landmarks):
            entity_type = np.zeros((5))
            if i == 0:
                entity_type[0] = 1
                landmark.type = 0
            elif i < 7:
                ty = (i-1) % 3 + 1
                entity_type[ty] = 1
                flag = False; pos = None
                while not flag:
                    flag = True
                    pos = self.get_new_respawn_position()
                    for j in range(1, i-1):
                        dist = np.sqrt(np.square(pos - world.landmarks[j].state.p_pos).sum())
                        if dist < 0.1:
                            flag = False
                            break
                landmark.state.p_pos = pos
                landmark.type = ty
                c = np.zeros((3,))
                c[ty-1] = 1
                landmark.color = c
            else:
                landmark.state.p_pos = self.get_new_invader_position()
                entity_type[-1] = 1
                landmark.type = 4
            entity_info = np.concatenate([landmark.state.p_pos] + [entity_type])
            world.other_entities.append(entity_info)

        self.fill_observability_mask(world)

    def fill_observability_mask(self, world):
        observable_range = world.observable_range
        max_n_agents = len(world.agents)
        max_n_entities = len(world.landmarks) + max_n_agents

        world.observability_mask.fill(0)
        #mask = np.zeros((max_n_agents, max_n_entities))
        #mask_alive = np.zeros((max_n_agents, max_n_entities))

        for i, agent in enumerate(world.agents):
            if not agent.alive:
                continue

            for j, a in enumerate(world.agents):
                if a.alive:
                    world.observability_mask[i,j+max_n_entities] = 1.
                    #mask_alive[i,j] = 1.
                    if np.sqrt(np.sum(np.square(agent.state.p_pos - a.state.p_pos))) <= observable_range:
                        #mask[i,j] = 1.
                        world.observability_mask[i,j] = 1.

            for j, lm in enumerate(world.landmarks):
                if lm.alive:
                    #mask_alive[i,max_n_agents+j] = 1.
                    world.observability_mask[i,max_n_agents+max_n_entities+j] =1.
                    if np.sqrt(np.sum(np.square(agent.state.p_pos - lm.state.p_pos))) <= (observable_range + lm.size):
                        #mask[i,max_n_agents+j] = 1.
                        world.observability_mask[i,max_n_agents+j] = 1.
        #world.observability_mask = np.concatenate([mask, mask_alive], -1)

    def get_new_invader_position(self):
        th = np.random.rand() * np.pi * 2
        xy = np.clip(np.array([np.cos(th), np.sin(th)]) * 1.42, -0.99, 0.99)
        return xy

    def get_new_respawn_position(self):
        theta = np.random.rand() * 2 * np.pi
        cos = np.cos(theta)
        sin = np.sin(theta)
        radius = np.random.rand() * 0.55 + 0.4
        x = max(min(cos * radius, 0.95), -0.95)
        y = max(min(sin * radius, 0.95), -0.95)
        pos = np.array([x, y])
        return pos

    def post_step(self, world):
        for i, lm in enumerate(world.landmarks):
            if i == 0:
                continue # skip home
            if not lm.alive: # respawn the landmark
                if i < 7:
                    lm.alive = True
                    flag = False; pos = None
                    while not flag:
                        flag = True
                        pos = self.get_new_respawn_position()
                        for j in range(1, 7):
                            if not world.landmarks[j].alive:
                                continue
                            dist = np.sqrt(np.square(pos - world.landmarks[j].state.p_pos).sum())
                            if dist < 0.1:
                                flag = False
                                break
                    lm.state.p_pos = pos
                else:
                    if self.invader_respawn_time <= 0:
                        lm.alive = True
                        lm.state.p_pos = self.get_new_invader_position()
                    else:
                        self.invader_respawn_time -= 1
            elif i == 7: # invader is getting closer to home
                lm.state.p_pos = lm.state.p_pos - lm.state.p_pos / np.sqrt(np.square(lm.state.p_pos).sum()) * 0.05

        for agent in world.agents:
            if not agent.alive:
                continue
            pos = agent.state.p_pos.copy()
            if agent.hold > 0:
                c = np.zeros((3,))
                c[agent.hold-1] = 1
                agent.color = c
            else:
                agent.color = agent.attribute[:3]

        for i, lm in enumerate(world.landmarks):
            world.other_entities[i][:2] = lm.state.p_pos

        self.fill_observability_mask(world)

    def is_collision(self, agent1, agent2, extra=0.01):
        dist = self.dist(agent1, agent2)
        dist_min = agent1.size + agent2.size
        return True if dist <= (dist_min + extra) else False

    def dist(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        return dist

    def reward(self, agent, world):
        if not agent.alive:
            return 0

        reward = 0.

        if world.landmarks[7].alive and self.is_collision(world.landmarks[0], world.landmarks[7]):
            reward -= 4
            world.landmarks[7].alive = False
            world.landmarks[7].state.p_pos = np.zeros((2,))
            self.invader_respawn_time = np.random.randint(10) + 1

        if world.landmarks[7].alive and self.is_collision(agent, world.landmarks[7]):
            reward += 4
            world.landmarks[7].alive = False
            world.landmarks[7].state.p_pos = np.zeros((2,))
            self.invader_respawn_time = np.random.randint(10) + 1
            agent.at = 4
            return reward

        for lm in world.landmarks[1:7]:
            if lm.alive and self.is_collision(agent, lm) and agent.hold == 0:
                lm.alive = False
                agent.hold = lm.type
                reward += agent.attribute[agent.hold-1] * 10
                agent.at = lm.type
                return reward

        if self.is_collision(agent, world.landmarks[0]) and agent.hold > 0:
            reward += 1.
            agent.hold = 0
            agent.at = 0
            return reward

        agent.at = -1
        return reward

    def global_reward(self, world):
        r = 0.
        for a in world.agents:
            r += self.reward(agent, world)
        return r

    def done(self, agent, world):
        return False

    def observation(self, agent, world):
        hold = np.zeros((4,))
        hold[agent.hold] = 1
        at = np.zeros((6,))
        at[agent.at+1] = 1
        # [2 + 2 + 4 + 5]
        o = np.concatenate(
                [agent.state.p_vel] + [agent.state.p_pos] + [hold] + [at])
        return o

def make_env(benchmark=False, discrete_action=True):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    scenario = Scenario()
    # create world
    world = scenario.make_world()

    # create multiagent environment
    if hasattr(scenario, 'post_step'):
        post_step = scenario.post_step
    else:
        post_step = None

    if hasattr(scenario, 'done'):
        done = scenario.done
    else:
        done = None

    if benchmark:
        env = MultiAgentEnv(world,
                            reset_callback=scenario.reset_world,
                            reward_callback=scenario.reward,
                            observation_callback=scenario.observation,
                            post_step_callback=post_step,
                            info_callback=scenario.benchmark_data,
                            done_callback=done)
    else:
        env = MultiAgentEnv(world,
                            reset_callback=scenario.reset_world,
                            reward_callback=scenario.reward,
                            observation_callback=scenario.observation,
                            post_step_callback=post_step,
                            done_callback=done)
    return env
