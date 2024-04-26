import random
import numpy as np
import gym
import tensorflow as tf
from collections import deque

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import gym_gomoku

tf.compat.v1.disable_eager_execution()

class MyAgent:
    def __init__(self, player_mark, board_size, action_size, learning_rate=0.001, observation_space=None):
        self.learning_rate = learning_rate
        self.state_size = (board_size, board_size, observation_space.shape[-1])
        self.action_size = action_size
        self.env = env
        self.observation_space = observation_space
        self.epsilon = 1.0
        self.min_epsilon = 0.01
        self.board_size = board_size
        self.player_mark = player_mark
        self.model = self.get_model()
        self.replay_buffer = []
        self.gamma = 0.95
        self.observation_space = observation_space
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        self.update_targetnn_rate = 10
        self.valid_moves = np.arange(self.board_size ** 2)
        self.main_network = self.get_model()
        self.target_network = self.get_model()

        self.target_network.set_weights(self.main_network.get_weights())

    def get_model(self):
        input_layer = Input(shape=self.state_size)
        x = Conv2D(32, (3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
        x = Conv2D(32, (3, 3), strides=(1, 1), padding='valid', activation='relu')(x)
        x = Flatten()(x)
        output_layer = Dense(self.action_size, activation='linear')(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def get_batch_from_buffer(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        indices = np.random.choice(len(self.replay_buffer), size=batch_size, replace=False)
        batch = np.zeros((batch_size, self.state_size[0], self.state_size[1], self.state_size[2] + 1))

        for i, idx in enumerate(indices):
            state, action, reward, next_state, done = self.replay_buffer[idx]
            batch[i, :, :, :-1] = state
            batch[i, :, :, -1] = action

        state_batch = batch[:, :, :, :-1]
        action_batch = batch[:, :, :, -1].astype(int).reshape(-1)
        reward_batch = np.array([self.replay_buffer[idx][2] for idx in indices])
        next_state_batch = np.array([self.replay_buffer[idx][3] for idx in indices])
        done_batch = np.array([self.replay_buffer[idx][4] for idx in indices], dtype=bool)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def train_main_network(self, batch_size):
        # Kiểm tra số lượng bộ dữ liệu trong replay_buffer
        if len(self.replay_buffer) < batch_size:
            return

        # Lấy một lô mẫu từ replay_buffer
        minibatch = random.sample(self.replay_buffer, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.get_batch_from_buffer(batch_size)

        # Cập nhật replay_buffer
        target_f = reward_batch + (1 - done_batch) * self.gamma * \
                     np.amax(self.target_network.predict(next_state_batch), axis=1)

        target = self.main_network.predict(state_batch)
        target[np.arange(batch_size), action_batch] = target_f
        self.main_network.fit(state_batch, target, epochs=1, verbose=0)

        # Cập nhật mạng target
        self.update_target_network()

    # Phương thức này sẽ trả về một danh sách các hành động hợp lệ trong trạng thái hiện tại    
    def get_q_value(self, state, action):
            valid_moves = self.get_legal_moves(state)
            if len(valid_moves) == 0:
                return 0

            action_idx = np.where((valid_moves == action).all(axis=1))[0]
            if len(action_idx) == 0:
                return 0
            
            expanded_state = np.expand_dims(state, axis=-1)
            expanded_state = np.repeat(expanded_state, repeats= self.state_size[-1], axis=-1)
            q_values = self.model.predict(np.expand_dims(expanded_state, axis=0)).ravel()
            
            return q_values[action_idx[0]]
    
    def update_q_value(self, state, action, next_state, next_action, reward, done):
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.get_q_value(next_state, next_action)

        td_error = td_target - self.get_q_value(state, action)

        valid_moves = self.get_legal_moves(state)
        if len(valid_moves) == 0:
            return
        action_idx = np.where((valid_moves == action).all(axis=1))[0]
        if len(action_idx) == 0:
            return
        q_values = self.model.predict(np.expand_dims(state, axis=0)).ravel()

        q_values[action_idx[0]] += self.learning_rate * td_error
        
        self.model.fit(np.expand_dims(state, axis=0), q_values.reshape(1, -1), epochs=1, verbose=0)


    def get_next_state(self, state, action):
        # action là một giá trị nguyên chỉ số của hành động được chọn
        i, j = np.unravel_index(int(action), (self.board_size, self.board_size))
        if state[i, j] != 0:
            # Handle special case
            return None

        next_state = np.copy(state)
        next_state[i, j] = self.player_mark
        return next_state
    def move(self, state, action):
        # Chuyển action thành toạ độ của ô trên bàn cờ
        i, j = np.unravel_index(int(action), (self.board_size, self.board_size))

        # Nếu ô đã được đánh rồi, trả về None
        if state[i, j] != 0:
            return None

        # Tạo trạng thái mới bằng cách đánh dấu ô tại vị trí (i, j) bằng player_mark
        next_state = np.copy(state)
        next_state[i, j] = self.player_mark
        return next_state
    def get_reward(self, state, next_state):
        # Check if the game has ended
        winner = self.check_winner(next_state)
        if winner is not None:
            if winner == self.player_mark:
                # The agent has won
                return 1
            else:
                # The agent has lost
                return -1

        # Compute the number of empty cells left in the next state
        empty_cells = len(np.where(next_state == 0)[0])

        # Compute the difference between the number of empty cells in the current state and the next state
        empty_cells_diff = empty_cells - len(np.where(state == 0)[0])

        # Return the difference as the reward
        return empty_cells_diff
    def check_winner(self, state):
        # Kiểm tra hàng ngang
        for i in range(self.board_size):
            for j in range(self.board_size - 4):
                row = state[i, j:j+5]
                if np.array_equal(row, np.array([1, 1, 1, 1, 1])):
                    return 1
                if np.array_equal(row, np.array([-1, -1, -1, -1, -1])):
                    return -1

        # Kiểm tra hàng dọc
        for i in range(self.board_size - 4):
            for j in range(self.board_size):
                col = state[i:i+5, j]
                if np.array_equal(col, np.array([1, 1, 1, 1, 1])):
                    return 1
                if np.array_equal(col, np.array([-1, -1, -1, -1, -1])):
                    return -1

        # Kiểm tra đường chéo chính
        for i in range(self.board_size - 4):
            for j in range(self.board_size - 4):
                diag = state[i:i+5, j:j+5].diagonal()
                if np.array_equal(diag, np.array([1, 1, 1, 1, 1])):
                    return 1
                if np.array_equal(diag, np.array([-1, -1, -1, -1, -1])):
                    return -1

        # Kiểm tra đường chéo phụ
        for i in range(4, self.board_size):
            for j in range(self.board_size - 4):
                anti_diag = np.fliplr(state)[i:i+5, j:j+5].diagonal()
                if np.array_equal(anti_diag, np.array([1, 1, 1, 1, 1])):
                    return 1
                if np.array_equal(anti_diag, np.array([-1, -1, -1, -1, -1])):
                    return -1

        # Không có cờ thủ nào thắng
        if np.all(state != 0):
            return 0

        # Trò chơi chưa kết thúc
        return None
    def get_legal_moves(self, state):
        positions = np.array(np.where(state == 0)).T
        legal_moves = [self.board_size * pos[0] + pos[1] for pos in positions]

        return legal_moves
    def is_valid_move(self, state, action):
        i, j = np.unravel_index(int(action), (self.board_size, self.board_size))
        return state[i, j] == 0
    def make_decision(self, state):
        if state is None:
            # Xử lý trường hợp đặc biệt
            return None, None
        print(state)
        # Lấy các nước đi hợp lệ từ hàm get_legal_moves(state)
        valid_moves = self.get_legal_moves(state)

        # Kiểm tra nếu không có nước đi hợp lệ:
        if not valid_moves:
            return None, None

        # Mở rộng trạng thái để phù hợp kích thước của mô hình
        expanded_state = np.expand_dims(state, axis=-1)
        expanded_state = np.repeat(expanded_state, repeats=self.state_size[-1], axis=-1)

        # Tính toán giá trị Q cho trạng thái hiện tại
        q_values = self.model.predict(np.expand_dims(expanded_state, axis=0)).ravel()

        if np.random.uniform() < self.epsilon:
            # Chọn một hành động ngẫu nhiên trong các hành động hợp lệ
            return np.random.choice(valid_moves), None
        else:
            # Lấy hành động có giá trị Q cao nhất
            valid_q_values = q_values[valid_moves]
            best_action_idx = np.argmax(valid_q_values)
            best_action = valid_moves[best_action_idx]
            next_state = self.move(state, best_action)

            return best_action, next_state

    def move(self, state, action):
        # Chuyển action thành toạ độ của ô trên bàn cờ
        i, j = np.unravel_index(int(action), (self.board_size, self.board_size))

        # Nếu ô đã được đánh rồi, trả về None
        if state[i, j] != 0:
            return None

        # Tạo trạng thái mới bằng cách đánh dấu ô tại vị trí (i, j) bằng player_mark
        next_state = np.copy(state)
        next_state[i, j] = self.player_mark
        
        # Kiểm tra trạng thái tiếp theo có hợp lệ không
        if next_state is None:
            return None
        else:
            return next_state
    def experience_replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        minibatch = random.sample(self.replay_buffer, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            if next_state is None:
                continue  # Skip invalid action in the environment
            
            # ...
            
            target_f = self.main_network.predict(np.expand_dims(state, axis=0), verbose=0)
            target_f[0][action] = target

            self.main_network.fit(np.expand_dims(state, axis=0), target_f, epochs=1, verbose=0)
    def update_epsilon(self, iteration):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay**iteration)

# Khởi tạo môi trường
env = gym.make('Gomoku9x9-v0')
observation_space = env.observation_space

state = env.reset()

# Định nghĩa state_size và action_size
state_size = env.observation_space.shape
action_size = env.action_space.n
# Định nghĩa tham số khác
n_episodes =6000
n_timesteps = 50
batch_size = 128

# Khởi tạo agent
board_size = 9 

my_agent = MyAgent(player_mark=1, board_size=board_size, action_size=action_size, learning_rate=0.001, observation_space=env.observation_space)
win_count = 0
win_rates = []
total_rewards = []

for i in range(n_episodes):
    done = False
    state = env.reset()
    episode_reward = 0
    step = 0
    while not done:
        # Lấy hành động của agent
        action, next_state = my_agent.make_decision(state)
        try:
            next_state, reward, done, _ = env.step(action)
        except gym.error.Error:
            # Chọn một hành động khác nếu hành động ban đầu không hợp lệ
            legal_moves = env.legal_moves
            if not legal_moves:
                break
            action = np.random.choice(legal_moves)
            next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        my_agent.remember(state, action, reward, next_state, done)
        state = next_state
        step += 1
        
        # Cập nhật mô hình
        if step % n_timesteps == 0:
            my_agent.train_main_network(batch_size)
            my_agent.experience_replay()

        # Cập nhật epsilon
        my_agent.update_epsilon(step)

    win_rates.append((win_count/(i+1))*100)
    total_rewards.append(episode_reward)
    print("Win rate in episode %d: %.2f%%" % (i+1, win_rates[-1]))
    print("Episode Reward in episode %d: %.2f" % (i+1, episode_reward))   

plt.figure(figsize=(10,4))
plt.subplot(121)
plt.plot(list(range(1, n_episodes+1)), win_rates)
plt.title("Win rate of agent over episodes")
plt.xlabel("Episode")
plt.ylabel("Win rate (%)")
plt.ylim([0, 100])

plt.subplot(122)
plt.plot(list(range(1, n_episodes+1)), total_rewards)
plt.title("Total reward of agent over episodes")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.show()

