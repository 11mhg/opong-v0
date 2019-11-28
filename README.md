# OPONG V0

A custom implementation using pygame of the game of Pong. A custom implementation was made in order to make a custom state space encoding instead of using the atari implementation.

## Install
To Install, clone the repository using 
```bash
git clone https://github.com/11mhg/opong-v0
```
and run the following in opong-v0/

```bash
pip3 install -e ./ --user
```

## Usage
In order to use this environment, use the typical gym environment make:

```python
import gym_opong
import gym

env = gym.make('opong-v0')
```
