# https://gym.openai.com/envs/SpaceInvaders-v0/

# 1. Evaluar un Environment con OpenAI Gym (Acciones aleatorias)
import gym
import random

# Environment: Space Invaders
env = gym.make('SpaceInvaders-v0')
height, width, channels = env.observation_space.shape
print("Observation space: ", env.observation_space)

# Actions
actions = env.action_space.n
action_meanings = env.unwrapped.get_action_meanings()
print("Action space: ", env.action_space)
print("Action meanings: ", action_meanings)

# Episodes
episodes = 20
for episode in range(episodes):
    state = env.reset()
    done = False
    score = 0

    # done=True cuando es destruido
    while not done:
        env.render()
        action = random.choice([0, 1, 2, 3, 4, 5])
        n_state, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode+1, score))
env.close()

