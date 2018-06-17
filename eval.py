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

use_cuda = torch.cuda.is_available()


env = gym.make('CartPole-v0')
dqn = DQN()
dqn.load_state_dict(torch.load("dqn1.pth")) 
if use_cuda:
    print ("use cuda")
    dqn.cuda()
    cudnn.bachmark = True
dqn.eval()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor



print ("Start test")
total = 0
for episode in range(100):
    state = env.reset()
    env._max_episode_steps = sys.maxsize
    score = 0
    while True:
        action = dqn(FloatTensor([state])).data.max(1)[1].view(1,1)

        next_state, reward, done, info = env.step(int(action[0]))
       # print(action,the_action)
        score += reward
        state = next_state
        if done:
            total += score
            break

print ("Average reward: {}".format(total/100))
