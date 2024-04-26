import gym
import gym_gomoku
# Khởi tạo môi trường
import keras
import numpy as np
from gym_gomoku.envs import GomokuEnv
env = GomokuEnv(player_color='black', opponent='beginner', board_size=15)
state= env.reset()
state_size = env.observation_space.shape[0]

# Load agent đã train
my_agent = keras.models.load_model("train_agent.h5")
n_timesteps = 500
total_reward = 0

for t in range(n_timesteps):
    env.render()
    # Lấy state hiện tại đưa vào predict
    state = state.reshape((1, state_size))
    q_values = my_agent.predict(state, verbose=0)
    max_q_values = np.argmax(q_values)

    # Action vào env và lấy thông số
    next_state, reward, terminal, _, _ = env.step(action=max_q_values)
    total_reward += reward
    state = next_state
    print(t)