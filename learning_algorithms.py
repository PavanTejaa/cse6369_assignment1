import pickle
import gymnasium as gym
import torch
from gymnasium.utils.save_video import save_video
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical
from utils import *


# Class for training an RL agent within an environment
class PGTrainer:
    def __init__(self, params):
        self.params = params
        self.env = gym.make(self.params['env_name'])
        self.agent = Agent(env=self.env, params=self.params)
        self.actor_policy = PGPolicy(input_size=self.env.observation_space.shape[0], output_size=self.env.action_space.n, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.optimizer = Adam(params=self.actor_policy.parameters(), lr=self.params['lr'])

    def run_training_loop(self):
        list_ro_reward = list()

        for ro_idx in range(self.params['n_rollout']):
            trajectory = self.agent.collect_trajectory(policy=self.actor_policy)
            loss = self.estimate_loss_function(trajectory)
            self.update_policy(loss)
            # TODO: Calculate avg reward for this rollout
            # HINT: Add all the rewards from each trajectory. There should be "ntr" trajectories within a single rollout.
            sum_of_rewards = 0
            reward_list = trajectory.get('reward')
            for trajectory_reward_list in reward_list:\
                sum_of_rewards += apply_return(trajectory_reward_list)
            avg_ro_reward = sum_of_rewards/len(reward_list)
            print(f'#################End of rollout {ro_idx}: Average trajectory reward is {avg_ro_reward: 0.2f}')
            # Append average rollout reward into a list
            list_ro_reward.append(avg_ro_reward)
        # Save avg-rewards as pickle files
        pkl_file_name = self.params['exp_name'] + '.pkl'
        with open(pkl_file_name, 'wb') as f:
            pickle.dump(list_ro_reward, f)
        # Save a video of the trained agent playing
        self.generate_video()
        # Close environment
        self.env.close()

    def estimate_loss_function(self, trajectory):
        loss = list()
        log_probabilities_list = trajectory.get('log_prob')
        reward_list = trajectory.get('reward')
        for t_idx in range(self.params['n_trajectory_per_rollout']):
            # TODO: Compute loss function
            # HINT 1: You should implement eq 6, 7 and 8 here. Which will be used based on the flags set from the main function
            
            # HINT 2: Get trajectory action log-prob
            
            # HINT 3: Calculate Loss function and append to the list
            log_probabilities_list_idx = log_probabilities_list[t_idx]
            reward_list_idx = reward_list[t_idx]
            # print('LOG LIST:',log_probabilities_list_idx)
            # print('Length of LOG:',len(log_probabilities_list_idx))
            # print('Reward LIST:',reward_list_idx)
            # print('Length of Reward:',len(reward_list_idx))
            if(self.params['reward_to_go'] == True):
                #print('----------------RTG----------------')
                reward_to_go_list = apply_reward_to_go(reward_list_idx)
                loss_idx = 0
                for idx in range(len(log_probabilities_list_idx)):
                    loss_idx = loss_idx + log_probabilities_list_idx[idx]*reward_to_go_list[idx]
                #print(reward_to_go_list)
                #print(log_probabilities_list_idx)
            elif(self.params['reward_discount'] == True):
                reward_discount_list = apply_discount(reward_list_idx)
                loss_idx = 0
                for idx in range(len(log_probabilities_list_idx)):
                    loss_idx = loss_idx + log_probabilities_list_idx[idx]*reward_discount_list[idx]
            else:
                reward_sum = apply_return(reward_list_idx)
                # log_probability_sum = 0
                # for log_prob_idx in log_probabilities_list_idx:
                #     log_probability_sum =+ log_prob_idx
                log_probability_sum = log_probabilities_list_idx.sum()
                # print('REWARD SUM: ',reward_sum)
                # print('LOG SUM: ',log_probability_sum)
                loss_idx = log_probability_sum*reward_sum
            loss.append(loss_idx*-1)
        #print(loss)
        loss = torch.stack(loss).mean()
        return loss

    def update_policy(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def generate_video(self, max_frame=1000):
        # Generating the video 5 times with random initial states
        for i in range(5):
            self.env = gym.make(self.params['env_name'], render_mode='rgb_array_list')
            obs, _ = self.env.reset()
            for _ in range(max_frame):
                action_idx, log_prob = self.actor_policy(torch.tensor(obs, dtype=torch.float32, device=get_device()))
                obs, reward, terminated, truncated, info = self.env.step(self.agent.action_space[action_idx.item()])
                print('Action Taken:', action_idx.item(),'& Reward: ', reward)
                if terminated or truncated:
                    break
            save_video(frames=self.env.render(), video_folder=self.params['env_name'][:-3], name_prefix=(self.params['exp_name'])+'_video'+str(i),fps=self.env.metadata['render_fps'], step_starting_index=1, episode_index=1)


# CLass for policy-net
class PGPolicy(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(PGPolicy, self).__init__()
        # TODO: Define the policy net
        # HINT: You can use nn.Sequential to set up a 2 layer feedforward neural network.
        # Initializing the feed forward neural network with one input layer, one hidden layer, one output layer and ending with softman function.
        self.policy_net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size),
            nn.Softmax()
        )

    def forward(self, obs):
        # TODO: Forward pass of policy net
        # HINT: (use Categorical from torch.distributions to draw samples and log-prob from model output)
        #print('OBS:',torch.from_numpy(obs))
        probabilities = self.policy_net(obs)
        distribution = Categorical(probabilities)
        action_index = distribution.sample()
        log_prob = distribution.log_prob(action_index)
        # print(probabilities)
        # print(action_index)
        # print(log_prob)
        # print('Action Taken:', action_index, ' & Log Probability: ', log_prob)
        return action_index, log_prob


# Class for agent
class Agent:
    def __init__(self, env, params=None):
        self.env = env
        self.params = params
        self.action_space = [action for action in range(self.env.action_space.n)]

    def collect_trajectory(self, policy):
        obs, _ = self.env.reset(seed=self.params['rng_seed'])
        rollout_buffer = list()
        for _ in range(self.params['n_trajectory_per_rollout']):
            trajectory_buffer = {'log_prob': list(), 'reward': list()}
            while True:
                # TODO: Get action from the policy (forward pass of policy net)
                action_idx, log_prob = policy.forward(torch.from_numpy(obs))
                # TODO: Step environment (use self.env.step() function)
                obs, reward, terminated, truncated, info = self.env.step(action_idx.item())
                print('Action Taken:', action_idx.item(), ' & Reward: ', reward)
                # Save log-prob and reward into the buffer
                trajectory_buffer['log_prob'].append(log_prob)
                trajectory_buffer['reward'].append(reward)
                # Check for termination criteria
                if terminated or truncated:
                    obs, _ = self.env.reset()
                    rollout_buffer.append(trajectory_buffer)
                    break
        rollout_buffer = self.serialize_trajectory(rollout_buffer)
        #print(rollout_buffer)
        return rollout_buffer

    # Converts a list-of-dictionary into dictionary-of-list
    @staticmethod
    def serialize_trajectory(rollout_buffer):
        serialized_buffer = {'log_prob': list(), 'reward': list()}
        for trajectory_buffer in rollout_buffer:
            serialized_buffer['log_prob'].append(torch.stack(trajectory_buffer['log_prob']))
            serialized_buffer['reward'].append(trajectory_buffer['reward'])
        return serialized_buffer

