import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# === Parameters ===
state_size = 19
action_size = 3

# === DQNetwork ===
class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 448),
            nn.ReLU(),
            nn.Linear(448, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# === Prioritized Replay Buffer ===
class PrioritizedReplayBuffer:
    def __init__(self, max_size=10000, alpha=0.8986914922994615):
        self.buffer = []
        self.priorities = []
        self.max_size = max_size
        self.alpha = alpha
        self.pos = 0

    def add(self, experience, td_error=1.0):
        priority = (abs(td_error) + 1e-6) ** self.alpha
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = experience
            self.priorities[self.pos] = priority
            self.pos = (self.pos + 1) % self.max_size

    def sample(self, batch_size, beta=0.592508933050511):
        if len(self.buffer) == 0:
            return [], [], [], [], [], [], []

        scaled_priorities = np.array(self.priorities)
        probs = scaled_priorities / scaled_priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = map(np.array, zip(*experiences))
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            weights
        )

    def update_priorities(self, indices, td_errors):
        for i, td in zip(indices, td_errors):
            self.priorities[i] = (abs(td) + 1e-6) ** self.alpha

    def size(self):
        return len(self.buffer)

# === Reward Function ===
def calculate_reward(pct_change, action, confidence, uncertainty,
                     reward_scale=12.73351139849768,
                     good_exp=1.3414311542877946,
                     bad_exp=1.3610432203028622,
                     no_trade_threshold=0.048565886119730874,
                     no_trade_factor=1.1361155731582184):
    movement = abs(pct_change)
    correct_direction = (
        (action == 2 and pct_change > 0) or
        (action == 0 and pct_change < 0)
    )
    if action == 1:
        if movement < no_trade_threshold:
            reward = movement ** good_exp * (1 - confidence) * no_trade_factor
        else:
            reward = -(movement ** bad_exp) * (1 - confidence) * no_trade_factor
    elif correct_direction:
        reward = (movement ** good_exp) * (1 + confidence)
    else:
        reward = -(movement ** bad_exp) * (1 + confidence)
    return float(np.clip(reward * reward_scale, -reward_scale, reward_scale))

# === Ensemble Prediction ===
def ensemble_predict(x, ensemble_models):
    preds = np.array([model.predict_proba(x)[:, 1] for model in ensemble_models])
    return preds.mean(), preds.std()

# === DQN Agent ===
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = DQNetwork(state_size, action_size)
        self.target_model = DQNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0008676626173731369)
        self.criterion = nn.MSELoss(reduction='none')
        self.buffer = PrioritizedReplayBuffer(alpha=0.8986914922994615)
        self.batch_size = 16
        self.gamma = 0.9600361938657662
        self.update_target_steps = 5
        self.step = 0
        self.epsilon_start = 0.5416568572943327
        self.epsilon_end = 0.08373408816385998
        self.epsilon_decay = 0.008544280861733894

    def get_epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-self.epsilon_decay * self.step)

    def act(self, state, y_proba, uncertainty):
        rand = np.random.rand()
        if y_proba > 0.6:
            return 2
        elif y_proba < 0.4:
            return 0
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor).detach().numpy().flatten()
        if rand < self.get_epsilon():
            return np.random.choice([0, 1, 2])
        else:
            exp_q = np.exp(q_values - np.max(q_values))
            probs = exp_q / exp_q.sum()
            return np.random.choice(self.action_size, p=probs)

    def remember(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        with torch.no_grad():
            target = reward + self.gamma * self.target_model(next_tensor).max(1)[0].item() * (1 - done)
            current = self.model(state_tensor)[0][action].item()
        td_error = target - current
        self.buffer.add((state, action, reward, next_state, done), td_error)

    def train(self):
        if self.buffer.size() < self.batch_size:
            return
        beta = min(1.0, 0.592508933050511 + self.step * 1e-4)
        s, a, r, ns, d, idxs, w = self.buffer.sample(self.batch_size, beta)
        s = torch.FloatTensor(s)
        a = torch.LongTensor(a)
        r = torch.FloatTensor(r)
        ns = torch.FloatTensor(ns)
        d = torch.FloatTensor(d)
        w = torch.FloatTensor(w)
        with torch.no_grad():
            next_a = self.model(ns).argmax(1, keepdim=True)
            next_q = self.target_model(ns).gather(1, next_a).squeeze(1)
            expected_q = r + (1 - d) * self.gamma * next_q
        current_q = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)
        td_errors = expected_q - current_q
        loss = (self.criterion(current_q, expected_q) * w).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.buffer.update_priorities(idxs, td_errors.detach().numpy())
        self.step += 1
        if self.step % self.update_target_steps == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step
        }, path)

    def load(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.step = checkpoint['step']
