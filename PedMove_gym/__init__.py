from gym.envs.registration import register

register(
    id='PedSimPred-v0',
    entry_point='PedMove_gym.envs:PedSimPred',
)