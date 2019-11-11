from gym.envs.registration import register

register(
        id='opong-v0',
        entry_point='gym_opong.envs:PongEnv'
)


