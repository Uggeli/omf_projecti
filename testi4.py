from pyautogui import getWindowsWithTitle, screenshot, press, keyDown, keyUp
import pyscreeze
import cv2
import numpy as np

from collections import deque, namedtuple

import math
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Enviroment2 import Enviroment
from model16 import Agent as Fighter

moves_list_p1 = [
    'up',
    'down',
    'left',
    'right',
    'enter',
    'shiftright',
    ''
]

moves_list_p2 = [
    'w',
    'x',
    'a',
    'd',
    'tab',
    'ctrlright',
    ''
]

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class Replay_memory:
    def __init__(self, mem_len):
        self.mem_len = mem_len
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.mem_len:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.mem_len

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def save_model():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    torch.save(critic.state_dict(),
               fr'G:\Kdaus\pyyttoni\Omf_projekti2\critic_models\{timestr}critic.tar')


def optimize():
    if len(player1_memory) < BATCH_SIZE:
        return

    p1transitions = player1_memory.sample(BATCH_SIZE)
    p1batch = Transition(*zip(*p1transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = player1(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = player1_target(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    player1_optim.zero_grad()
    loss.backward()
    for param in player1.parameters():
        param.grad.data.clamp_(-1, 1)
    player1_optim.step()


def stack_frames():
    for _ in range(4):
        cur_screen = capture_screen()
        img_stack.append(screen_preprocess(cur_screen))

env = Enviroment()
player1_memory = Replay_memory(1_000_000)
player2_memory = Replay_memory(1_000_000)

LEARNING_RATE = 0.009
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
steps_done = 0
device = 'cpu'


player1 = Fighter(4, 7, 'Fighter', 4, device).to(device)
player1_target = Fighter(4, 7, 'Fighter', 4, device).to(device)
player1_target.load_state_dict(player1.state_dict())
player1_optim = optim.Adagrad(player1.parameters(), lr=LEARNING_RATE)

img_stack = deque([env.screen_preprocess(env.capture_screen()) for
                  _ in range(4)],
                  maxlen=4)

last_screen = torch.tensor(img_stack).float().to(device).view(320, 4, 220)
current_screen = torch.tensor(img_stack).float().to(device).view(320, 4, 220)
state = current_screen - last_screen

p1_last_scores = deque([1], maxlen=100)


def select_action(player, state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return player(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(7)]], device=device, dtype=torch.long)


training = True
while training:
    if env.match() == 'fight':
        torch.cuda.empty_cache()
        in_fight = True
        gameover = False

        p1_last = 196
        p2_last = 196

        p1_total_score = []
        p2_total_score = []

        player1_target.load_state_dict(player1.state_dict())
        # player2_target.load_state_dict(player2.state_dict())

        while in_fight:
            if env.match() != 'fight':
                in_fight = False
                break

            if gameover:
                print('!GAME OVER!')
                p1_total_score.clear()
                p2_total_score.clear()
                break
            try:
                p1_action = select_action(player1, state)
                press(moves_list_p1[p1_action])
            except:
                pass

            p1, p2 = env.get_fight_status()
            if p1 >= 200:
                p2_score = 100
                p1 = p1_last
                gameover = True
            else:
                p2_score = p1_last - p1
                p1_last = p1

            if p2 >= 200:
                p1_score = 100
                p2 = p2_last
                gameover = True
            else:
                p1_score = p2_last - p2
                p2_last = p2

            # p1_last_scores.append(p1_score)
            # p2_last_scores.append(p2_score)

            # if sum(p1_last_scores) == 0:
            #     p1_score = -1.

            # if sum(p2_last_scores) == 0:
            #     p2_score = -1.

            print(f'P1:{p1} score:{p1_score} move:{moves_list_p1[p1_action]}')
            # print(f'p2:{p2} score:{p2_score} move:{moves_list_p2[p2_action]}')

            last_screen = current_screen
            img_stack.append(env.screen_preprocess(env.capture_screen()))
            current_screen = torch.tensor(img_stack).float().unsqueeze(0).to(device).view(320, 4, 220)

            if gameover:
                print('!GAME OVER!')
                p1_total_score.clear()
                p2_total_score.clear()
                next_state = None
            else:
                next_state = current_screen - last_screen

            player1_memory.push(state, p1_action, next_state, torch.tensor([p1_score], device=device))
            # player2_memory.push(state, p2_action, next_state, torch.tensor([p2_score], device=device))

            state = next_state
            optimize()

            if gameover:
                break

            p1_total_score.append(p1_score)
            p2_total_score.append(p2_score)

    else:
        press(['enter', 'ctrlright'])