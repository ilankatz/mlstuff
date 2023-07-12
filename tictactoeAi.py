import numpy as np
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)  # 3x3 game board
        self.current_player = 1  # 1 for X, -1 for O
        self.game_over = False
        self.winner = None

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.game_over = False

    def get_valid_moves(self):
        return np.argwhere(self.board == 0)  # Return empty cells as valid moves

    def make_move(self, move):
        move = tuple(move)  # Convert move to a tuple if it's not already
        if self.board[move[0], move[1]] == 0:
            self.board[move[0], move[1]] = self.current_player
            self.current_player *= -1  # Switch players
            self.check_game_over()


    def check_game_over(self):
        for i in range(3):
            if abs(np.sum(self.board[i, :])) == 3 or abs(np.sum(self.board[:, i])) == 3:
                self.game_over = True
                self.winner = self.current_player
                return

        if abs(np.sum(np.diag(self.board))) == 3 or abs(np.sum(np.diag(np.fliplr(self.board)))) == 3:
            self.game_over = True
            self.winner = self.current_player
            return

    # Check for a draw
        if len(self.get_valid_moves()) == 0:
            self.game_over = True
            self.winner = 0  # Draw


    def get_state(self):
        return self.board.flatten().tolist()

    def print_board(self):
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        for row in self.board:
            print("|".join([symbols[cell] for cell in row]))
            print("-----")
        print()

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the RL agent using Q-learning
class RLAgent:
    def __init__(self, input_size, output_size):
        self.policy_net = DQN(input_size, output_size)
        self.target_net = DQN(input_size, output_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.01)
        self.gamma = 0.9 
        self.memory = []  
        self.batch_size = 128
        self.current_player = 1

    def choose_action(self, state, valid_moves):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.policy_net(state)
        
        valid_q_values = [q_values[i] for i in range(len(valid_moves))]
        max_q_value = max(valid_q_values)
        max_indices = [i for i, q_value in enumerate(valid_q_values) if q_value == max_q_value]
        action_index = random.choice(max_indices)
        return action_index

    def update_q_network(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch = zip(*batch)

        state_batch = torch.tensor(state_batch, dtype=torch.float32)
        action_batch = torch.tensor(action_batch, dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32)

        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_memory(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
# Train the agent
env = TicTacToe()
input_size = 9
output_size = 9
agent = RLAgent(input_size, output_size)
total_episodes = 10000

try:
    for episode in range(total_episodes):
        if episode %100 == 0:
            print(episode)
        env.reset()
        state = env.get_state()

        while not env.game_over:
            valid_moves = env.get_valid_moves()
            action = agent.choose_action(state, valid_moves)  # Include valid_moves argument
            env.make_move((valid_moves[action]))

            next_state = env.get_state()

            if env.game_over:
                reward = env.winner
            else:
                reward = 0

            agent.update_memory(state, action, reward, next_state)
            agent.update_q_network()
            agent.update_target_network()

            state = next_state

except KeyboardInterrupt:
    print("Keyboard interrupt received. Saving the agent...")
    model_path = "rl_agent_interrupt.pkl"
    with open(model_path, "wb") as file:
        pickle.dump(agent, file)
    print("Agent saved successfully.")


# Test the agent against a human player
model_path = "rl_ttt_model7.pkl"
with open(model_path, "wb") as file:
    pickle.dump(agent, file)
env.reset()
state = env.get_state()

while not env.game_over:
    env.print_board()

    if env.current_player == 1:
        valid_moves = env.get_valid_moves()
        action_index = agent.choose_action(state, valid_moves)
        action = valid_moves[action_index]
        env.make_move(action)
    else:
        valid_moves = env.get_valid_moves()
        print("Valid moves:", valid_moves)
        row = int(input("Enter the row (0-2): "))
        col = int(input("Enter the column (0-2): "))
        action = (row, col)

        # Check if the entered action is valid
        if action not in valid_moves:
            print("Invalid move. Please try again.")
            continue

        env.make_move(action)

    state = env.get_state()

    if env.game_over:
        env.print_board()
        if env.winner == 1:
            print("Human wins!")
        elif env.winner == -1:
            print("Agent wins!")
        elif env.winner == 0:
            print("It's a draw!")

env.print_board()
