import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from pathlib import Path
from datetime import datetime
from collections import namedtuple, deque
import random
from copy import deepcopy, copy
from tqdm.notebook import tqdm


class experienceReplayBuffer:
    '''Class for the experience replay buffer'''
    def __init__(self, memory_size=5000, burn_in=1000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'next_state'])
        self.replay_memory = deque(maxlen=memory_size)

    def sample_batch(self, batch_size=32, device='cpu'):
        '''Sample a batch of experiences from the replay memory and return it as a tuple of tensors'''
        samples = random.sample(self.replay_memory, k=batch_size)
        states = torch.FloatTensor(np.vstack([e.state for e in samples])).to(device)
        actions = torch.LongTensor(np.vstack([e.action for e in samples])).to(device)
        rewards = torch.FloatTensor(np.vstack([e.reward for e in samples])).to(device)
        dones = torch.ByteTensor(np.vstack([e.done for e in samples]).astype(np.uint8)).to(device)
        next_states = torch.FloatTensor(np.vstack([e.next_state for e in samples])).to(device)
        return (states, actions, rewards, dones, next_states)

    def append(self, state, action, reward, done, next_state):
        '''Add a new experience to memory.'''
        exp = self.experience(state, action, reward, done, next_state)
        self.replay_memory.append(exp)

    def burn_in_capacity(self):
        return len(self.replay_memory) / self.burn_in
    

class NoisyLinear(nn.Linear):
    ''' Noisy linear module for NoisyNet'''
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.sigma_init = sigma_init
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.Tensor(out_features))
            self.register_buffer('epsilon_bias', torch.zeros(out_features))
        self.reset_parameters()


    def reset_parameters(self):
        '''
        Initialize the biases and weights
        '''
        if hasattr(self, 'sigma_bias'):
            nn.init.constant_(self.sigma_bias, self.sigma_init)
            nn.init.constant_(self.sigma_weight, self.sigma_init)

        std = np.sqrt(3/self.in_features)
        nn.init.uniform_(self.weight, -std, std)
        nn.init.uniform_(self.bias, -std, std)


    def forward(self, input):
        if self.bias is not None:
            ## NB: in place operation. PyTorch is not happy with that!! CHANGE IT
            self.epsilon_bias.data.normal_()

            bias = self.bias + self.sigma_bias*self.epsilon_bias
        else:
            bias = self.bias

        ## NB: in place operation. PyTorch is not happy with that!! CHANGE IT
        self.epsilon_weight.data.normal_()
        # new weight with noise
        weight = self.weight + self.sigma_weight*self.epsilon_weight
        # create the linear layer it the added noise
        return F.linear(input, weight, bias)
    
    
class Qnet(nn.Module):
    '''Class for the DQN neural network'''
    def __init__(self, n_inputs, n_hidden, n_outputs, noisyNet=False, dueling=False):
        super().__init__()
        """
        Params
        ======
        n_inputs: number of inputs
        n_hl1: number of units in the first hidden layer
        n_hl2: number of units in the second hidden layer
        n_outputs: number of outputs
        noisyNet: boolean to use NoisyNet
        duel: boolean to use Dueling DQN
        """
        self.n_inputs = n_inputs
        self.n_hl1 = n_hidden
        self.n_hl2 = n_hidden
        self.n_outputs = n_outputs
        self.noisyNet = noisyNet
        self.dueling = dueling
        
        # Selects if wer use NoisyNet or not
        linearLayer = NoisyLinear if self.noisyNet else nn.Linear
        
        # Input and first hidden layer
        self.fc1 = linearLayer(self.n_inputs, self.n_hl1)
        self.fc2 = linearLayer(self.n_hl1, self.n_hl2)
        
        # Second hidden layer and output layer
        if self.dueling:
            self.advantage = linearLayer(self.n_hl2, self.n_outputs)
            self.value = linearLayer(self.n_hl2, 1)
        else:
            self.fc3 = linearLayer(self.n_hl2, self.n_outputs)


    def forward(self, x):
        '''Forward pass through the network'''
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.dueling:
            value = self.value(x)
            advantage = self.advantage(x)
            return value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            return self.fc3(x)
    
    
    def get_qvals(self, state):
        '''Returns the q-values for a given state'''
        if state.dim == 1:
            state = torch.unsqueeze(state, 0)
        return self.forward(state)
    
    
class DQNAgent:
    '''Class for the DQN agent'''
    def __init__(self, env, device=torch.device('cpu'), gamma=0.99, lr=0.001, n_hidden=64, 
                 mem_size=8000, double=False, dueling=False, noisyNet=False):
        """"
        Params
        ======
        env: environment
        network: neural network
        buffer: clase con el buffer de repetición de experiencias
        epsilon: epsilon
        eps_decay: epsilon decay
        eps_min: minimum epsilon
        batch_size: batch size
        nblock: bloque de los X últimos episodios de los que se calculará la media de recompensa
        device: dispositivo
        actions: array de acciones posibles
        """

        self.env = env
        self.device = device
        
        # DQN versions
        self.double = double
        self.dueling = dueling
        self.noisyNet = noisyNet
        
        # Create main and target networks and send them to the device
        self.main_network = Qnet(self.env.observation_space.shape[0], n_hidden, env.action_space.n, 
                                noisyNet=self.noisyNet, dueling=self.dueling).to(self.device)
        self.target_network = Qnet(self.env.observation_space.shape[0], n_hidden, env.action_space.n,
                                  noisyNet=self.noisyNet, dueling=self.dueling).to(self.device)
        # Create the buffer
        self.buffer = experienceReplayBuffer(memory_size=mem_size, burn_in=1000)
        
        # Set hyperparameters
        self.gamma = gamma
        self.nblock = 100
        
        # Create the writer
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.name = 'DQN_'+'_'.join([name for name, val in 
                              zip(['double', 'dueling', 'noisy'], [self.double, self.dueling, self.noisyNet])  if val])
        self.writer = SummaryWriter(Path('runs') / self.name / self.timestamp)
        self.writer.add_graph(self.main_network, torch.Tensor(env.reset()[0]).to(self.device).unsqueeze(0))
        
        # Action space
        self.actions = [i for i in range(self.env.action_space.n)]
        
        # Set optimizer for the main network
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=lr)
        
        # Initialize
        self.update_loss = []
        self.total_reward = 0
        self.step_count = 0
        
    
    ### e-greedy method
    def get_action(self, state, epsilon=0.0):
        '''Return an action using an epsilon-greedy policy if not noisyNet, otherwise use a greedy policy.
             Params
            ======
            state (array_like): current state
            epsilon (float): epsilon, for epsilon-greedy action selection
        '''
        
        if not self.noisyNet and (np.random.random() < epsilon):
            action = np.random.choice(self.actions)
        else:
            state = torch.Tensor(state).unsqueeze(0).to(self.device)
            self.main_network.eval()
            with torch.inference_mode():
                qvals = self.main_network(state)
            self.main_network.train()
            action= torch.argmax(qvals, dim=-1).cpu().item()
        return action


    def take_step(self, state, eps, mode='train'):
        # Get random action
        if mode == 'explore':
            action = self.env.action_space.sample()  
        # Get action from q-values
        else:
            action = self.get_action(state, eps)
            self.step_count += 1

        # Take step in environment
        new_state, reward, done, _, _ = self.env.step(action)
        self.total_reward += reward
        
        # Save experience in buffer
        self.buffer.append(state, action, reward, done, new_state) 
        return new_state, done


    def train(self, max_episodes=5000, min_episodios=250, 
              batch_size=32, eps_start=1.0, eps_end=0.01, eps_decay=0.995,
              dnn_update_frequency=4, dnn_sync_frequency=1000,
              reward_threshold=None) -> dict:
        
        self.batch_size = batch_size
        
        # Select reward threshold
        if not reward_threshold:
            reward_threshold = self.env.spec.reward_threshold
            
        # Initialize epsilon
        epsilon = eps_start
        
        # Initialize lists for tracking progress
        train_rewards = []
        train_mean_rewards = []
        train_epsilons = []
        train_losses = []
        
        # Fill the replay buffer with random experiences
        state = self.env.reset()[0]
        while self.buffer.burn_in_capacity() < 1:
            state, _ = self.take_step(state, epsilon, mode='explore')
        
        # Training loop
        episode, training = 0, True
        pbar = tqdm(desc='Training', total=max_episodes)
        
        while training:
            state = self.env.reset()[0]
            self.total_reward = 0
            gamedone = False
            pbar.update(1)
            
            # Play an episode
            for t in range(self.env.spec.max_episode_steps):
                # Take action in train mode (epsilon-greedy policy)
                state, gamedone = self.take_step(state, epsilon, mode='train')
                # Update the main network
                if not self.step_count % dnn_update_frequency:
                    self.update()
                # Target and main network syncronization
                if not self.step_count % dnn_sync_frequency:
                    self.target_network.load_state_dict(
                        self.main_network.state_dict())
                if gamedone:
                    break

            episode += 1
            
            # Save epsilon, loss, training reward and mean training reward
            train_rewards.append(self.total_reward)
            train_epsilons.append(epsilon)
            ep_loss = np.mean(np.array(self.update_loss))
            mean_rewards = np.mean(train_rewards[-self.nblock:])
            train_losses.append(ep_loss)
            train_mean_rewards.append(mean_rewards)
            
            # Add to tensorboard
            if self.writer is not None:
                self.writer.add_scalar("epsilon", epsilon, episode)
                self.writer.add_scalar("loss", ep_loss, episode)
                self.writer.add_scalar("reward", self.total_reward, episode)
                self.writer.add_scalar("mean_reward", mean_rewards, episode)
            
            self.update_loss = []

            print(f"\rEpisode {episode} Mean Rewards {mean_rewards:.2f} Epsilon {epsilon:.2f}\t\t", end="")

            # Check if the episode limit has been reached
            if episode >= max_episodes:
                training = False
                print('Episode limit reached.')
                break

            # Finishes the game if the average reward has reached the threshold set for this game and
            # a minimum number of episodes have been played
            if mean_rewards >= reward_threshold and min_episodios < episode:
                training = False
                print(f'Environment solved in {episode} episodes!')
                break

            # Epsilon update
            epsilon = max(epsilon * eps_decay, eps_end)
            
        training_stats = {
            'training_epsilons': np.array(train_epsilons),
            'training_rewards': np.array(train_rewards),
            'mean_training_rewards': np.array(train_mean_rewards),
            'training_losses': np.array(train_losses)
        }
            
        return training_stats


    def calculate_loss(self, batch):
        # We extract the states, actions, rewards, dones and next_states from the batch
        states, actions, rewards, dones, next_states = batch
        # Get q-values for all actions in current states
        currentQ = torch.gather(self.main_network.get_qvals(states), 1, actions)
        # Get max q-values for next states from target network
        nextQ = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        # Q-vals to 0 in terminal states
        nextQ[dones] = 0
        # Bellman equation
        if self.double:
            max_nextQ = torch.max(nextQ, dim=1)[0]
            max_nextQ = max_nextQ.view(max_nextQ.size(0), 1)
            expectedQ = self.gamma * max_nextQ + rewards
        else:
            expectedQ = self.gamma * nextQ + rewards
        # Loss calculation
        loss = F.mse_loss(currentQ, expectedQ.reshape(-1,1))
        return loss


    def update(self):
        # Reset gradients
        self.optimizer.zero_grad()
        # Take a random batch from the buffer
        batch = self.buffer.sample_batch(batch_size=self.batch_size, device=self.device)
        # Calculate loss
        loss = self.calculate_loss(batch)
        # Backpropagate loss
        loss.backward()
        # Update network weights with gradients
        self.optimizer.step()
        # Save loss
        self.update_loss.append(loss.detach().to('cpu').item())