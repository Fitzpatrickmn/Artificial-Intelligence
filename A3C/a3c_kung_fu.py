import cv2
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributions as distributions
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
import tqdm
import glob
import io
import base64
import imageio
from IPython.display import HTML, display
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

### CREATE NEURAL NETWORK ARCHITECTURE ### 

class Network(nn.Module):

  def __init__(self, action_size):
    super(Network, self).__init__()
    # Convolutional Layers
    self.conv1 = torch.nn.Conv2d(in_channels = 4,  out_channels = 32, kernel_size = (3,3), stride = 2) #input = stack of 4 grayscale frames from the kungfu environment
    self.conv2 = torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), stride = 2)
    self.conv3 = torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), stride = 2)
    # Flatten Layer
    self.flatten = torch.nn.Flatten()
    # Fully Connected Layers
    self.fc1  = torch.nn.Linear(512, 128)
    self.fc2a = torch.nn.Linear(128, action_size) #Q-Value Outputs (Actions)
    self.fc2s = torch.nn.Linear(128, 1) #V(s) State Output (Critic)

  # Forward Propagation
  def forward(self, state):
    x = self.conv1(state)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = self.conv3(x)
    x = F.relu(x)
    x = self.flatten(x)
    x = self.fc1(x)
    x = F.relu(x)
    action_values = self.fc2a(x)
    state_value = self.fc2s(x).squeeze()
    return action_values, state_value

### SET UP TO TRAIN THE AI ###

class PreprocessAtari(ObservationWrapper):

  def __init__(self, env, height = 42, width = 42, crop = lambda img: img, dim_order = 'pytorch', color = False, n_frames = 4):
    # Object Variables
    super(PreprocessAtari, self).__init__(env)
    self.img_size = (height, width)
    self.crop = crop
    self.dim_order = dim_order
    self.color = color
    self.frame_stack = n_frames
    n_channels = 3 * n_frames if color else n_frames
    obs_shape = {'tensorflow': (height, width, n_channels), 'pytorch': (n_channels, height, width)}[dim_order]
    self.observation_space = Box(0.0, 1.0, obs_shape)
    self.frames = np.zeros(obs_shape, dtype = np.float32)

  def reset(self):
    # Reset Environment
    self.frames = np.zeros_like(self.frames)
    obs, info = self.env.reset()
    self.update_buffer(obs)
    return self.frames, info

  def observation(self, img):
    # Preprocess Images of Environment
    img = self.crop(img)
    img = cv2.resize(img, self.img_size)
    if not self.color:
      if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype('float32') / 255.
    if self.color:
      self.frames = np.roll(self.frames, shift = -3, axis = 0)
    else:
      self.frames = np.roll(self.frames, shift = -1, axis = 0)
    if self.color:
      self.frames[-3:] = img
    else:
      self.frames[-1] = img
    return self.frames

  def update_buffer(self, obs):
    # Update Buffer
    self.frames = self.observation(obs)

def make_env():
  # Create the Environment
  env = gym.make("KungFuMasterDeterministic-v0", render_mode = 'rgb_array')
  env = PreprocessAtari(env, height = 42, width = 42, crop = lambda img: img, dim_order = 'pytorch', color = False, n_frames = 4)
  return env

env = make_env()

state_shape = env.observation_space.shape
number_actions = env.action_space.n
print("State shape:", state_shape)
print("Number actions:", number_actions)
print("Action names:", env.env.env.get_action_meanings())

# Initialize Hyperparameters
learning_rate = 1e-4
discount_factor = 0.99
number_environments = 10

### CREATE AGENT CLASS ###

class Agent():

  def __init__(self, action_size):
    # Object Variables
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.action_size = action_size
    self.network = Network(action_size).to(self.device)
    self.optimizer = torch.optim.Adam(self.network.parameters(), lr = learning_rate)
    #self.epsilon = epsilon

  ### USE SOFTMAX ###
  def act(self, state):
    # Actions Given a State
    if state.ndim == 3: # ensure our state is in a batch
      state = [state]
    state = torch.tensor(state, dtype = torch.float32, device = self.device)
    action_values, _ = self.network(state)
    policy = F.softmax(action_values, dim = -1) #softmax = action selection policy
    #return an array of random indices for each entry in the array based on output probabilities derived by softmax
    return np.array([np.random.choice(len(p), p = p) for p in policy.detach().cpu().numpy()])

  ### USE EPSILON GREEDY ### --> Commented out for future optimization
  #def act(self, state):
    #if state.ndim == 3:
      #state = [state]
    #state = torch.tensor(state, dtype=torch.float32, device=self.device)
    #action_values, _ = self.network(state)

    ## Epsilon-greedy exploration
    #if np.random.rand() < self.epsilon:
       # Randomly select an action
      #actions = np.random.randint(self.action_size, size=state.shape[0])
    #else:
       # Select actions greedily based on Q-values
      #policy = F.softmax(action_values, dim=-1)
      #actions = policy.max(1)[1].cpu().numpy()

    #return actions

  def step(self, state, action, reward, next_state, done):
    # called when agent takes a step in the environemnt
    # receives information such as current_state, action taken, reward, next_state & whether the episode is done
    # updates the models parameters/weights to take better actions that lead to a higher score
    batch_size = state.shape[0]
    # convert arrays into torch tensors
    state = torch.tensor(state, dtype = torch.float32, device = self.device)
    next_state = torch.tensor(next_state, dtype = torch.float32, device = self.device)
    reward = torch.tensor(reward, dtype = torch.float32, device = self.device)
    done = torch.tensor(done, dtype = torch.bool, device = self.device).to(dtype = torch.float32)
    # obtain action, state, & next_state values
    action_values, state_value = self.network(state)
    _, next_state_value = self.network(next_state)
    # calculate target state value via bellman equation
    target_state_value = reward + discount_factor * next_state_value * (1 - done)
    # calculate the advantage
    advantage = target_state_value - state_value
    # obtain probability, log probability, & entropy in order to calculate actor, critic, & total loss
    probs = F.softmax(action_values, dim = -1)
    logprobs = F.log_softmax(action_values, dim = -1)
    entropy = -torch.sum(probs * logprobs, axis = -1)
    batch_idx = np.arange(batch_size)
    logp_actions = logprobs[batch_idx, action] # log probability of the actions that were taken
    actor_loss = -(logp_actions * advantage.detach()).mean() - 0.001 * entropy.mean()
    critic_loss = F.mse_loss(target_state_value.detach(), state_value)
    total_loss = actor_loss + critic_loss
    # reset the optimizer
    self.optimizer.zero_grad()
    # back propagate total_loss
    total_loss.backward()
    # use the optimizer to update the weights and minimize total loss
    self.optimizer.step()


# Initialize Agent
agent = Agent(number_actions)

# Evaluate Agent 
def evaluate(agent, env, n_episodes = 1):
  episodes_rewards = []
  for _ in range(n_episodes):
    #initialize the state
    state, _ = env.reset()
    total_reward = 0
    while True:
      # play an action
      action = agent.act(state)
      # obtain state, reward, done boolean & information
      state, reward, done, info, _ = env.step(action[0])
      # calculate total reward
      total_reward += reward
      if done: # break the loop if the episode is done
        break
    # update episode rewards
    episodes_rewards.append(total_reward)
  # return total rewards for each episode
  return episodes_rewards
  print(episodes_rewards)

### MANAGE SIMULTANEOUS ENVIRONMENTS ###

class EnvBatch:
  # Parallelize reinforcement learning

  def __init__(self, n_envs = 10):
    self.envs = [make_env() for _ in range(n_envs)]

  def reset(self):
    # returns an array of initiliazed states
    _states = []
    for env in self.envs:
      _states.append(env.reset()[0])
    return np.array(_states)

  def step(self, actions):
    # function to take steps in multiple environements and return the relevant information
    next_states, rewards, dones, info, _ = map(np.array, zip(*[env.step(a) for env, a in zip(self.envs, actions)]))
    # check if the environment has finished, before returning relevant information
    for i in range(len(self.envs)):
      if dones[i]:
        next_states[i] = self.envs[i].reset()[0]
    return next_states, rewards, dones, info

# Train the Agent
env_batch = EnvBatch(number_environments)
batch_states = env_batch.reset()

with tqdm.trange(0, 3001) as progress_bar:
  # use tqdm to visualize training
  for i in progress_bar:
    batch_actions = agent.act(batch_states)
    batch_next_states, batch_rewards, batch_dones, _ = env_batch.step(batch_actions)
    batch_rewards *= 0.01 #stabilize training by reducing the magnitude of the rewards
    agent.step(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)
    batch_states = batch_next_states #set up for next training iteration
    if i % 1000 == 0:
      print("Average agent reward: ", np.mean(evaluate(agent, env, n_episodes = 10)))

### VISUALIZE RESULTS ###

def show_video_of_model(agent, env):
  state, _ = env.reset()
  done = False
  frames = []
  while not done:
    frame = env.render()
    frames.append(frame)
    action = agent.act(state)
    state, reward, done, _, _ = env.step(action[0])
  env.close()
  imageio.mimsave('A3C_Kungfu_03212024.mp4', frames, fps=30)

show_video_of_model(agent, env)

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