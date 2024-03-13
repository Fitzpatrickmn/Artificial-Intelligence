import os
import random
import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from collections import deque, namedtuple
import gymnasium as gym
import glob
import io
import base64
import imageio
from IPython.display import HTML, display
from gym.wrappers.monitoring.video_recorder import VideoRecorder

### BUILD THE AI ###

# Create the architecture of the Neural Network
class Network(nn.Module):
  # Initialize the network layout
  def __init__(self, state_size, action_size, seed = 42):
    super(Network, self).__init__()
    # Activate the seed
    self.seed = torch.manual_seed(seed)
    #First fully connected layer
    self.fc1 = nn.Linear(state_size, 64)
    # Second fully connected layer
    self.fc2 = nn.Linear(64, 64)
    # Third fully connected layer
    self.fc3 = nn.Linear(64, action_size)

  #Forward propagate the signal from input layer
  def forward(self, state):
    x = self.fc1(state)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    return self.fc3(x)

### TRAIN THE AI

# Set up the environment
env = gym.make('LunarLander-v2')
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
#print('State shape: ', state_shape)
#print('State size: ', state_size)
#print('Number of actions: ', action_size)

# Initialize the hyperparameters 
learning_rate = 5e-4
minibatch_size = 100
discount_factor = 0.99
replay_buffer_size = int(1e5)
interpolation_parameter = 1e-3

# Implement Experience Replay
class ReplayMemory(object):

  def __init__(self, capacity):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.capacity = capacity
    self.memory = []

  # Add new events into memory; delete oldest memory if capacity is at threshold
  def push(self, event):
    self.memory.append(event)
    if len(self.memory) > self.capacity:
      del self.memory[0]

  # Randomly select a batch of experiences from memory
  def sample(self, batch_size):
    experiences = random.sample(self.memory, k = batch_size)
    # Convert arrays into torch tensors
    states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
    actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
    rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
    next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
    dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
    return states, next_states, actions, rewards, dones

### BUILD OUT DQN ### 

# Define the behavior of the Agent 
class Agent():
  def __init__(self, state_size, action_size):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.state_size = state_size
    self.action_size = action_size
    # Maintains 2 q-networks (local and target) to stabilize learning
    self.local_qnetwork = Network(state_size, action_size).to(self.device) #Local Network: selects actions
    self.target_qnetwork = Network(state_size, action_size).to(self.device) #Target Network: calculates target q-values used in training the local q-network
    self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr = learning_rate)
    self.memory = ReplayMemory(replay_buffer_size)
    self.time_step = 0

  # Store experiences and decide when to learn from them
  def step(self, state, action, reward, next_state, done):
    self.memory.push((state, action, reward, next_state, done))
    self.time_step = (self.time_step + 1) % 4  # increment by 1; reset every 4 steps
    if self.time_step == 0:
      if len(self.memory.memory) > minibatch_size:
        experiences = self.memory.sample(100)
        self.learn(experiences, discount_factor)

  # Select action based on given state
  def act(self, state, epsilon = 0.):
    state = torch.from_numpy(state).float().unsqueeze(0).to(self.device) #convert state to tensor & add dimension: batch
    self.local_qnetwork.eval() #evaluation mode
    with torch.no_grad(): #ensure you are in inference mode, not training mode
      action_values = self.local_qnetwork(state) #get action values via local predictions
    self.local_qnetwork.train() #return to training mode
    # Generate random number, if larger than epsilon, use action with highest Q value, else, select random action
    if random.random() > epsilon:
      return np.argmax(action_values.cpu().data.numpy())
    else:
      return random.choice(np.arange(self.action_size))

  # Learn from experiences
  def learn(self, experiences, discount_factor):
    states, next_states, actions, rewards, dones = experiences
    next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1) #max q values for next_state
    q_targets = rewards + discount_factor * next_q_targets * (1 - dones) #q targets for current state from target q network
    q_expected = self.local_qnetwork(states).gather(1, actions) #expected q values from local q network
    loss = F.mse_loss(q_expected, q_targets) #loss between expected and target q values (mean squared error loss)
    self.optimizer.zero_grad() #reset the optimizer
    loss.backward() #backpropagaion of loss to compute gradient with respect to model parameters
    self.optimizer.step() #single optimization step to update model parameters
    self.soft_update(self.local_qnetwork, self.target_qnetwork, interpolation_parameter) #update target network parameters with local parameters

  # Gradual adjustment of model parameters via soft update
  def soft_update(self, local_model, target_model, interpolation_parameter):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()): #loop through target and local parameters
      target_param.data.copy_(interpolation_parameter * local_param.data + (1.0 - interpolation_parameter) * target_param.data) #updating target parameters

# Initialize the DQN agent
agent = Agent(state_size, action_size)

### TRAIN THE DQN AGENT ###

# Set up training hyperparameters
number_episodes = 2000
maximum_number_timesteps_per_episode = 1000
epsilon_start_value  = 1.0
epsilon_end_value  = 0.01
epsilon_decay_value  = 0.995
epsilon = epsilon_start_value
scores_on_100_episodes = deque(maxlen = 100)

# Initialize training
for episode in range(1, number_episodes + 1):
  state, _ = env.reset() # reset to initial state
  score = 0
  for t in range(maximum_number_timesteps_per_episode):
    action = agent.act(state, epsilon)
    next_state, reward, done, _, _ = env.step(action)
    agent.step(state, action, reward, next_state, done)
    state = next_state
    score += reward
    if done:
      break
  scores_on_100_episodes.append(score) #append score of latest episode
  epsilon = max(epsilon_end_value, epsilon_decay_value * epsilon)

  # Print results
  print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)), end = "")
  if episode % 100 == 0:
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)))
  if np.mean(scores_on_100_episodes) >= 200.0:
    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100, np.mean(scores_on_100_episodes)))
    torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
    break

### VISUALIZE RESULTS ###

# Generate and display video of agent in the LunarLander environment 
def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action.item())
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)

show_video_of_model(agent, 'LunarLander-v2')

# Display video stored as MP4
def show_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

show_video()