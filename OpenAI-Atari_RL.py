# ==================================
# Crear un modelo de RL para el juego
# ===================================
import warnings
warnings.filterwarnings('ignore')

import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# Environment: Space Invaders
env = gym.make('SpaceInvaders-v0')
# Observaciones
height, width, channels = env.observation_space.shape
# Acciones
actions = env.action_space.n


# Crear un modelo DL como base del agente
def build_model(height, width, channels, actions):
    cnn_model = Sequential()
    cnn_model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(3, height, width, channels)))
    cnn_model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    cnn_model.add(Convolution2D(64, (3, 3), activation='relu'))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(512, activation='relu'))
    cnn_model.add(Dense(256, activation='relu'))
    cnn_model.add(Dense(actions, activation='linear'))
    return cnn_model


model = build_model(height, width, channels, actions)
model.summary()


# Construyendo el Agente con Keras-RL
# https://keras-rl.readthedocs.io/en/latest/agents/overview/
# Deep-Q Network
# https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy


def build_agent(model, actions, policy):
    dqn = DQNAgent(model=model, policy=policy, nb_actions=actions, memory=SequentialMemory(limit=1000, window_length=3),
                   enable_dueling_network=True, dueling_type='avg', nb_steps_warmup=1000)
    return dqn


# Policy
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=10000)
# Agent
agent = build_agent(model, actions, policy)


# Train the model
agent.compile(Adam(lr=1e-4))
agent.fit(env, nb_steps=10000, visualize=False, verbose=2)


# Save the model
agent.save_weights('model/dqn_atari_agent_weights.h5f')

# =============================
# Cargar el modelo RL entrenado
# =============================

# Load the model
model_name = 'dqn_atari_agent_weights.h5f'
agent.load_weights('model/' + model_name)
print("Model {} loaded ...".format(model_name))

# Test de model
scores = agent.test(env, nb_episodes=5, visualize=True)
print(np.mean(scores.history['episode_reward']))