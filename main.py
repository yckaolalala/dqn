import gym
import random
import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.utils.data
import torch.backends.cudnn  as cudnn
from six.moves import cPickle
import sys

# hyper parameters
EPISODES = 3000  
GAMMA = 0.95  
LR = 0.0005  
batch_size = 128  
epsilon = 1
eps_end = 0.01  
eps_decay = 0.995  
capacity = 5000
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(4, 32)
        self.l2 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def epsilon_greedy(state):
    global epsilon
    sample = random.random()
    epsilon =  max(eps_end, epsilon * eps_decay)
    if sample > epsilon:
        return dqn(state).detach().data.max(1)[1].view(1,1)

    else:
        return LongTensor([[random.randrange(2)]])


use_cuda = torch.cuda.is_available()
env = gym.make('CartPole-v0')
dqn = DQN()
if use_cuda:
    print ("use cuda")
    dqn.cuda()
    cudnn.bachmark = True
dqn.train()
memory = ReplayMemory(capacity)
optimizer = optim.Adam(dqn.parameters(), LR)
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor


    
step = 0
histroy = {}
total = 0
_max = 0
for episode in range(EPISODES):
    state = env.reset()
    env._max_episode_steps = sys.maxsize
    score = 0
    while True:
        action = epsilon_greedy(FloatTensor([state]))
        next_state, reward, done, info = env.step(int(action[0]))
        step += 1
        score += 1
        memory.push((state, action, next_state, reward, done))
        state = next_state
        if step % 50 == 0:
            if total > _max:
                _max = total
                print ("save best network at episode {}".format(episode))
                torch.save(dqn.state_dict(), "dqn_best.pth")
            total = 0
            for i in range(50):
                if len(memory) >= batch_size:
                    transitions = memory.sample(batch_size)
                    batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*transitions)
    
                    cur_q = dqn(FloatTensor(batch_state)).gather(1, torch.cat(batch_action))
                    max_next_q = dqn(FloatTensor(batch_next_state)).detach().max(1)[0]
                    y = FloatTensor(batch_reward).view(-1,1)
                    for idx, row in enumerate(batch_done):
                        if row == False:
                            y[idx] += GAMMA * max_next_q[idx]
                    loss = F.mse_loss(cur_q, y.view(-1,1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
        if done:
            print("Episode {}: score {} ".format(episode,score+1))
            break
    total += score
    histroy[episode] = score
    with open('history_log2.pkl', 'wb') as f:
        cPickle.dump(histroy, f)
    torch.save(dqn.state_dict(), "dqn_episode.pth")

