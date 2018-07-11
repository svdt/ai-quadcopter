import sys
import numpy as np
from wrapper import envWrapper
from drone import DroneEnv

TESTING = 0
CONTINUE = 0
if '-test' in sys.argv:
    TESTING = 1
if '-continue' in sys.argv:
    CONTINUE = 1

RENDER = 0
FRAMESKIP = 2
OUTPUT_GRAPH = True
LOG_DIR = './log'
SAVE_DIR = './model/model.ckpt'
MAX_GLOBAL_EP = 24000
UPDATE_ITER = 5
GAMMA = 0.95
ENTROPY_BETA = 0.005
LR_AC = 1e-4    # learning rate for actor
KEEP_N = 0
max_grad_norm = 1e3
ALPHA = 0.99
EPS = 1e-5

if TESTING:
    RENDER = True
    FRAMESKIP = 1
    MAX_GLOBAL_EP = 10000

EPISODE = 0
CONTINUOUS = 1

# import roboschool
import gym
from gym.envs.mujoco.pusher import PusherEnv
# envClass = 'RoboschoolInvertedPendulum-v1'
envClass = DroneEnv
# envClass = PusherEnv
if type(envClass) == str:
    env = gym.make(envClass)
else:
    env = envClass()
# env = envClass()
observation_space = env.observation_space
action_space = env.action_space
# A_BOUND = (action_space.low, action_space.high)
del env

agents = {
"lander": {"init_n": 16, "env": envClass},
}

if TESTING:
    for agent in agents:
        agents[agent]["init_n"] = 1
env = envWrapper(agents)
