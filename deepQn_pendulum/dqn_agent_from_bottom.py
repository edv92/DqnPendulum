import torch
import torch.nn as nn
import torch.nn.functional as F   # function to add convolutional layers
import torch.optim as optim
from Pendulum_cart import CartPendulum
from dataclasses import dataclass
from typing import Any
from math import pi
from collections import namedtuple, deque
import numpy as np
import random
import matplotlib.pyplot as plt
import wandb
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
@dataclass
class simulation_sample:
    state: Any
    action: int
    reward: float
    next_state: Any
"""
class Q_Agent:
    def __init__(self):
        None
    def get_eps_dec(self, epsilon_start, episode_end):
        eps_decay = epsilon_start/episode_end
        return eps_decay
    def update_epsilon(self, current_eps, eps_decay):
        new_eps = current_eps-eps_decay
        if new_eps <= 0:
            new_eps = 0
        return new_eps

    def update_epsilon_exponential(self, current_eps, eps_decay):
        new_eps = current_eps*eps_decay
        return new_eps

    def update_epsilon_exponential_v2(self,epsilon, eps_decay,episode):
        new_eps = epsilon*eps_decay**episode
        if new_eps>0.1:
            return new_eps
        return 0.1

    def get_next_action(self, epsilon, policy_net, state):
        #state = tuple(state)
        #print(type(state))
        state = torch.from_numpy(state).float()
        #print(state)
        #state = torch.tensor(state, device=device, dtype=torch.float)
        if np.random.random() > epsilon:
            with torch.no_grad():
                #print(torch.argmax(policy_net(state)) )
                #print(f"the output of network is {policy_net(state)}")
                #print(f"this is  net action tensor {torch.tensor([[torch.argmax(policy_net(state))]]) }")
                #return policy_net(state).max(1)[1].view(1, 1)
                return torch.tensor([[torch.argmax(policy_net(state))]], device=device, dtype=torch.long)
        else:
            #print(f"this is random action tensor {torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)}")
            return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)







class NeuralNetworkz(nn.Module):
    def __init__(self,num_observations,num_actions, num_neurons_fc_nn):
        super().__init__()
        self.num_obs = num_observations
        self.num_actions = num_actions

        self.net = nn.Sequential(
            nn.Linear(num_observations, num_neurons_fc_nn),
            nn.ReLU(),
            nn.Linear(num_neurons_fc_nn, num_neurons_fc_nn),
            #nn.ReLU(),
            nn.Linear(num_neurons_fc_nn, num_actions)
        )


    def forward(self, x):
        x = x.to(device)
        return self.net(x)



# Named tuple class to hold the current sample in simulation
sample = namedtuple('sample', ('state','action', 'reward', 'next_state'))

#holds samples of  (s, a ,r ,s')
class ReplayBuffer(object):
    def __init__(self,  buffer_size):
        #self.buffer = deque([], maxlen = buffer_size)
        self.buffer = [None]*buffer_size
        self.idx = 0
        self.buffer_size = buffer_size
    def insert_sample(self, *args):
        #self.buffer.append(sample(*args))

        self.buffer[self.idx % self.buffer_size] = sample(*args)

        self.idx +=1
        #print(self.buffer)
        #print(self.buffer)
    def get_samples(self, num_samples):
        assert num_samples < min(self.idx, self.buffer_size)
        return random.sample(self.buffer, num_samples)

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    wandb.init(project='dqn_pendulum', entity='edvard')
    #os.environ['WANDB_MODE'] = 'offline'
    # Environment parameters
    m2 = 0.1  # [kg]
    m1 = 1.0  # [kg]
    l = 1.0  # [m]
    kt = 0.03  # [kg/rad^2]

    # Applief force (abs value)
    applied_f = 10

    # Step size (time) for runge kutta
    dt = 0.02  # [s]

    # Borders for reward and simulation, theta
    theta_lim_sim = 21
    theta_lim_deg = 90
    theta_max = theta_lim_deg * (pi / 180)

    # X borders for reward and simulation
    x_lim = 1.5
    x_lim_sim = 1.6

    num_observations = 4
    num_actions = 2
    num_neurons = 200

    #objects
    p_obj = CartPendulum(m2, m1, l, kt)
    agent_obj = Q_Agent()
    policy_net = NeuralNetworkz(num_observations, num_actions, num_neurons).to(device)
    target_net = NeuralNetworkz(num_observations,num_actions,num_neurons).to(device)

    target_net.load_state_dict(policy_net.state_dict()) # copy the weights of policy net
    target_net.eval() #just to turn off training mode for the target network
    buffer_size = 200000
    buffer_obj = ReplayBuffer(buffer_size) #buffer size


    #Hyperparams
    batch_size = 30000
    gamma = 0.9999
    eps_const = 0.9
    epsilon = 0.9
    eps_decay = 0.99999
    learning_rate = 0.0005
    learning_rate_decay = 0.9
    l = learning_rate
    #eps_decay = agent_obj.get_eps_dec(1,200_000)
    target_update = 400000
    num_steps_before_train = 1000

    opt = optim.Adam(policy_net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size = 500000, gamma = learning_rate_decay)
    num_episodes = 2_000_000
    num_steps = 2000
    loss_arr = []
    steps_arr = np.zeros(num_episodes)
    episode_arr = np.arange(1,num_episodes)
    current_loss = 0
    avg_steps = 0
    total_reward = 0
    train_steps = 10
    target_steps = 0

    def train_model(buffer_size):
        if buffer_obj.idx< buffer_size:
            return

        samples = buffer_obj.get_samples(batch_size)
        batch = sample(*zip(*samples))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        #print(reward_batch)
        #print(action_batch)
        #print(next_state_batch)

        opt.zero_grad()
        q_values = policy_net(state_batch).gather(1, action_batch)
        #q_values = torch.reshape(q_values, (-1,))
        #detach so no gradient is involved in the target network
        next_q_values = target_net(next_state_batch).max(1)[0].detach()

        expected_q_values = (next_q_values*gamma)+reward_batch
        criterion = nn.SmoothL1Loss()
        loss = criterion(q_values, expected_q_values.unsqueeze(1))
        #loss = ((expected_q_values-q_values)**2).mean()
        loss.backward()
        opt.step()
        scheduler.step()

        return loss


    step_count = 0
    for episode in range(num_episodes):


        current_obs = np.array([pi,0,0,0])
        #


        for steps in range(num_steps):

            train_steps +=1
            target_steps +=1
            step_count +=1
            action = agent_obj.get_next_action(epsilon,policy_net,current_obs)
            #print(action)
            reward = p_obj.getReward(current_obs[0], theta_max, current_obs[1], x_lim, 0.1, -0.4)
            total_reward += reward



            f = p_obj.getForce(action,applied_f)


            next_obs = p_obj.sim_one_step_RK4(current_obs[0], current_obs[1], current_obs[2], current_obs[3], f, dt)


            buffer_obj.insert_sample(torch.from_numpy(np.array([current_obs])).float(), action, torch.tensor([reward], device=device, dtype=torch.float),
                                     torch.from_numpy(np.array([next_obs])).float())


            if step_count>= 500_000:
                l *= learning_rate_decay
                step_count = 0

            #train model
            if train_steps >= num_steps_before_train:
                train_steps = 0
                current_loss = train_model(buffer_size)
                wandb.log({'loss': current_loss}, step=episode)


                #loss_arr.append(current_loss)



            #update target_model
            if target_steps >= target_update:
                target_steps=0
                target_net.load_state_dict(policy_net.state_dict())
                torch.save(target_net.state_dict(),f"models14/{episode}.pth")
                print(f"learning rate:{l}")
            # Update state transition
            current_obs = next_obs
            """
            if next_obs[0] * (180 / pi) <= (-1 * theta_lim_sim) or next_obs[0] * (
                    180 / pi) >= theta_lim_sim:
                #print("was broken")
                break
            """
            if next_obs[1] >= x_lim_sim or next_obs[1] <= -x_lim_sim:
                break

        if episode % 1000 == 0:


            wandb.log({'epsilon': epsilon}, step=episode)
            wandb.log({'steps': step_count}, step = episode)
        wandb.log({'reward': total_reward}, step=episode)
        total_reward = 0
        #avg_steps+= step_count
        #steps_arr[episode] = step_count
        #update epsilon
        #epsilon  = agent_obj.update_epsilon(epsilon, eps_decay)
        epsilon = agent_obj.update_epsilon_exponential_v2(eps_const, eps_decay, episode)


            #print(f" for episode: {episode}, the step count is: {step_count}")
            #print(f"average num_steps: {avg_steps/target_update}")
            #print(f"current loss is: {current_loss}")
            #print(f"current epsilon is: {epsilon}")
            #avg_steps = 0

    #torch.save(policy_net.state_dict(), "C:\\Users\\edvar\\Documents\\Master Thesis\\deepQn_pendulum\\finished_model")


    """
    step_arr = np.arange(0,len(loss_arr))


    fig,ax =  plt.subplots()
    fig.suptitle("loss function per episode")
    ax.plot(step_arr, loss_arr)
    plt.show()

    sfig, sax = plt.subplots()
    sfig.suptitle("num_steps")
    sax.plot(step_arr, steps_arr)
    plt.show()
    #print(buffer_obj.buffer[1])
    #print(buffer_obj.buffer[2])
    #net_test = net(torch.Tensor(observations))
        #
    #print(net_test)
    """