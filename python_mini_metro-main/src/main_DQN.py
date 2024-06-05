import pygame

from config import framerate, screen_color, screen_height, screen_width,start_with_3_initial_paths
from event.convert import convert_pygame_event
from mediator import Mediator
from dqn import DQL_GCN, ReplayMemory, Transition
import json
import datetime
import time
import random

import math
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

# normal pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# graph pytorch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED=123
BATCH_SIZE = 128
NUM_EPISODES = 600
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.05
EPS_DECAY = NUM_EPISODES * 3
WARM_UP_STEPS = 1000 # number of timesteps before training starts
TAU = 0.0005
LR = 1e-4
INSPECT_FINAL_SOLUTION = False

# Get number of actions from action space
N_ACTIONS = 24
NODE_ATTRIBUTES = 7
NUM_STATIONS = 10


policy_net = DQL_GCN(NODE_ATTRIBUTES, N_ACTIONS, NUM_STATIONS).to(device)
target_net = DQL_GCN(NODE_ATTRIBUTES, N_ACTIONS, NUM_STATIONS).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(20000)


steps_done = 0


def select_action(state, mask):
    global steps_done
    final = False
    sample = random.random()
    if steps_done < WARM_UP_STEPS:
        eps_threshold = 1
    else:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * (steps_done-WARM_UP_STEPS) / EPS_DECAY)
    steps_done += 1

    available_actions = mediator.available_actions(mask=False)
    if len(available_actions) == 1:
        print("------------ FINAL ACTION ------------")
        final = True

    if sample > eps_threshold:
        with torch.no_grad():

            # mask output such that unavailable actions are replaced with -inf
            #mask = torch.tensor(mediator.available_actions(mask=True), dtype=torch.bool)
            output = policy_net(state['red'], state['blue'], state['green'], mask, batch_size=1)
            #print(f"Q values: {output}")
            #print(f"Mask: {mask}")


            output = output.masked_fill(~mask.unsqueeze(0), -float('inf'))
            #print(f"Masked Q values: {output}")
            #print(f"Masked Q values: {output}")
            #print(f"Max Q value: {output.max(1).indices.view(1, 1)} with value {output.max(1)}")

            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return output.max(1).indices.view(1, 1), final
    else:
        random_action = random.choice(available_actions)
        return torch.tensor([[random_action]], device=device, dtype=torch.long), final


episode_durations = []


def plot_durations(show_result=False, is_ipython=True, lookback=100):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= lookback:
        means = durations_t.unfold(0, lookback, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(lookback-1), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    # if is_ipython:
    #     if not show_result:
    #         display.display(plt.gcf())
    #         display.clear_output(wait=True)
    #     else:
    #         display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    state_batch_red = DataLoader([b['red'] for b in batch.state], batch_size=BATCH_SIZE, shuffle=False)
    state_batch_blue = DataLoader([b['blue'] for b in batch.state], batch_size=BATCH_SIZE, shuffle=False)
    state_batch_green = DataLoader([b['green'] for b in batch.state], batch_size=BATCH_SIZE, shuffle=False)
    mask_batch = torch.cat(batch.mask)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # extract batch from each color
    state_batch_red = next(iter(state_batch_red)).to(device)
    state_batch_blue = next(iter(state_batch_blue)).to(device)
    state_batch_green = next(iter(state_batch_green)).to(device)


    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net

    state_action_values = policy_net(state_batch_red, state_batch_blue, state_batch_green, mask_batch, batch_size=BATCH_SIZE).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)

    non_final_next_states_red = DataLoader([s['red'] for s in batch.next_state if s is not None], batch_size=sum(non_final_mask).item(), shuffle=False)
    non_final_next_states_blue = DataLoader([s['blue'] for s in batch.next_state if s is not None], batch_size=sum(non_final_mask).item(), shuffle=False)
    non_final_next_states_green = DataLoader([s['green'] for s in batch.next_state if s is not None], batch_size=sum(non_final_mask).item(), shuffle=False)
    non_final_next_mask = torch.cat([m for m in batch.next_mask if m is not None]) #CHECK OM DEN RETURNERER NONE I FINAL STATES (step_to_end)

    non_final_next_states_red = next(iter(non_final_next_states_red)).to(device)
    non_final_next_states_blue = next(iter(non_final_next_states_blue)).to(device)
    non_final_next_states_green = next(iter(non_final_next_states_green)).to(device)


    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states_red, non_final_next_states_blue, non_final_next_states_green, non_final_next_mask , batch_size=sum(non_final_mask).item()).max(1).values
    
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # debug
    n_actions = len(mediator.available_actions(mask=False))
    print(f"Loss: {loss.item()}, Actions: {n_actions}, Epsilon: {EPS_END + (EPS_START - EPS_END) * math.exp(-1. * (steps_done - WARM_UP_STEPS) / EPS_DECAY)}")

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()






####### MAIN LOOP ########

if torch.cuda.is_available():
    NUM_EPISODES = 600


for i_episode in range(NUM_EPISODES):
    
    # Initialize the environment and get its state
    print()
    print("*"*50)
    print(f"Initializing new episode: {i_episode} / {NUM_EPISODES}")
    pygame.init()
    flags = pygame.SCALED

    # game constants initialization
    #screen = pygame.display.set_mode((screen_width, screen_height), flags, vsync=1)

    
    mediator = Mediator(seed=SEED)
    mediator.initialize_with_3_paths(seed=SEED)
    mediator.initialize_action_mapping()

    # reset random
    random.seed(None)

    # screen.fill(screen_color)
    # mediator.render(screen)
    

    state, mask = mediator.state()


    # display
    #for event in pygame.event.get():
    #    pygame.display.flip()

    # Training loop
    #state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action, final = select_action(state, mask)
        print(f"Action: {action.item()}")
        if final:
            observation, reward, terminated = mediator.step_to_end(action.item(), reward_type='score')
        else:
            observation, reward, terminated = mediator.step(action.item(), reward_type='score')
        reward = torch.tensor([reward], device=device)
        done = terminated 

        if terminated:
            next_state = None
            next_mask = None
            
            if INSPECT_FINAL_SOLUTION:
            # display final setup
                screen.fill(screen_color)
                mediator.render(screen)
                pygame.event.get()
                pygame.display.flip()
                time.sleep(4)

        else:
            next_state, next_mask = observation

        # Store the transition in memory
        memory.push(state, mask, action, next_state, next_mask, reward)

        # Move to the next state
        state = next_state
        mask = next_mask

        # Perform one step of the optimization (on the policy network)
        #print(f"t={t} : Optimizing policy network")
        if steps_done > WARM_UP_STEPS:
            optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)

            plot_durations()
            print(f"Terminal state reached at {t} steps, final score: {mediator.score}")
            print("Episode finished")
            print("*"*50)
            print()
            
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()


