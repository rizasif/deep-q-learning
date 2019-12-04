import gym
from wimblepong.wimblepong import Wimblepong
from wimblepong.simple_ai import SimpleAi
# from wimblepong.a2c.agent import Agent
# from wimblepong.a2c.models import PolicyConv

# gym.envs.register("WimblepongVisualMultiplayer-v0",
#         entry_point="wimblepong:Wimblepong",
#         max_episode_steps=None,
#         kwargs={"opponent": None, "visual": True})

gym.envs.register("WimblepongVisualSimpleAI-v0",
        entry_point="wimblepong:Wimblepong",
        max_episode_steps=None,
        kwargs={"opponent": SimpleAi, "visual": True})

# gym.envs.register("WimblepongMultiplayer-v0",
#         entry_point="wimblepong:Wimblepong",
#         max_episode_steps=None,
#         kwargs={"opponent": None, "visual": False})

# gym.envs.register("WimblepongSimpleAI-v0",
#         entry_point="wimblepong:Wimblepong",
#         max_episode_steps=None,
#         kwargs={"opponent": SimpleAi, "visual": False})

