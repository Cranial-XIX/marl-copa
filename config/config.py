import argparse
import yaml

class Config:
    def __init__(self):
        with open("config/default.yaml", 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(self, key, value)
        self.has_coach = False
        self.init_args()

    def init_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--method')
        parser.add_argument('--centralized_every', type=int)
        parser.add_argument('--seed', type=int)
        parser.add_argument('--vi_lambda', type=float, default=0.001)
        parser.add_argument('--agent_hidden_dim', type=int, default=128)
        parser.add_argument('--env_name', type=str, default="mpe84")

        args = parser.parse_args()
        if args.method:
            self.method = args.method
        if args.centralized_every:
            self.centralized_every = args.centralized_every
        if args.seed:
            self.seed = args.seed
        self.vi_lambda = args.vi_lambda
        self.env_name = args.env_name
        self.agent_hidden_dim = args.agent_hidden_dim
        if "coach" in self.method:
            self.batch_size = int(self.batch_size * 8 / self.centralized_every)
        self.has_coach = "coach" in self.method

    def pprint(self):
        print("="*80)
        for k, v in self.__dict__.items():
            print(f"{str(k):20s}: {str(v):20s}")
        print("="*80)
