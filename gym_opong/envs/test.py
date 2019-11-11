import gym
import gym_opong


env = gym.make('opong-v0',enable_render=True)
env.reset()
while True:
    env.render()
    obs, r, done, _ = env.step(env.action_space.sample())
    if done:
        break




