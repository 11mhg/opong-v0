import gym
import gym_opong


env = gym.make('opong-v0',enable_render=True,draw_grid=True)
env.reset()
while True:
    env.render()
    action = env.action_space.sample()
    obs, r, done, _ = env.step(action)
    if done:
        break




