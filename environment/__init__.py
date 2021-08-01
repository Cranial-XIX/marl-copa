from gym.envs.registration import register

register(
    id='HuntSale-v0',
    entry_point='environment.huntsale:HuntSaleEnv',
)
