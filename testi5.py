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

moves_list_p1 = [
    'up',
    'down',
    'left',
    'right',
    'enter',
    'shiftright'
]

moves_list_p2 = [
    'w',
    'x',
    'a',
    'd',
    'tab',
    'ctrlright'
]

human_readable_action_list = [
    'up',
    'down',
    'left',
    'right',
    'punch',
    'kick'
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


class Fighter(nn.Module):
    def __init__(self):
        super(Fighter, self).__init__()
        self.steps_done = 0
        hidden = 128

        self.conv_layer = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=7, stride=2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=7, stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=7, stride=2),
            nn.BatchNorm2d(32)
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32)
        )

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(192, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, 6)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.conv_layer2(x)
        # # x = self.conv_layer3(x)
        # x = torch.flatten(x)
        x = self.output_layer(x.view(x.size(0), -1))

        return x


def save_model(model):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    torch.save(model.state_dict(),
               fr'G:\Kdaus\pyyttoni\Omf_projekti2\fighter_models\{timestr}Fighter.tar')


def optimize(agent, target, optim, memory):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                  batch.next_state)),
                                  device=device,
                                  dtype=torch.uint8)

    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    state_action_values = agent(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target(
                        non_final_next_states.to(device)).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values,
                            expected_state_action_values.unsqueeze(1))

    optim.zero_grad()
    loss.backward()
    optim.step()


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
device = 'cuda'


player1 = Fighter().to(device)
player1.load_state_dict(torch.load(r'G:\Kdaus\pyyttoni\Omf_projekti2\fighter_models\20190930-193442Fighter.tar'))
player1_target = Fighter().to(device)
player1_target.load_state_dict(player1.state_dict())
player1_optim = optim.Adagrad(player1.parameters(), lr=LEARNING_RATE)

player2 = Fighter().to(device)
player2.load_state_dict(torch.load(r'G:\Kdaus\pyyttoni\Omf_projekti2\fighter_models\20190930-193442Fighter.tar'))
player2_target = Fighter().to(device)
player2_target.load_state_dict(player2.state_dict())
player2_optim = optim.Adagrad(player2.parameters(), lr=LEARNING_RATE)

img_stack = deque([env.screen_preprocess(env.capture_screen()) for
                  _ in range(4)],
                  maxlen=4)

last_screen = torch.tensor(img_stack).float().unsqueeze(0).to(device)
current_screen = torch.tensor(img_stack).float().unsqueeze(0).to(device)
state = current_screen - last_screen

p1_last_scores = deque([1], maxlen=100)
p2_last_scores = deque([1], maxlen=100)


def select_action(player, state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * player.steps_done / EPS_DECAY)
    player.steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return player(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(6)]], device=device, dtype=torch.long)


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
        player2_target.load_state_dict(player2.state_dict())

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
                p2_action = select_action(player2, state)

                press((moves_list_p1[p1_action], moves_list_p2[p2_action]))
            except:
                pass

            p1, p2 = env.get_fight_status()
            if p1 >= 200:
                p2_score = 100.
                p1 = p1_last
                save_model(player2)
                gameover = True
            else:
                p2_score = float(p1_last - p1)
                p1_last = p1

            if p2 >= 200:
                p1_score = 100.
                p2 = p2_last
                save_model(player1)
                gameover = True
            else:
                p1_score = float(p2_last - p2)
                p2_last = p2

            p1_last_scores.append(p1_score)
            p2_last_scores.append(p2_score)

            if sum(p1_last_scores) == 0:
                p1_score = -1.

            if sum(p2_last_scores) == 0:
                p2_score = -1.

            print(f'P1:{p1} score:{p1_score}\
                move:{human_readable_action_list[p1_action]}\
                    p2:{p2} score:{p2_score}\
                        move:{human_readable_action_list[p2_action]}')

            last_screen = current_screen
            img_stack.append(env.screen_preprocess(env.capture_screen()))
            current_screen = torch.tensor(
                                img_stack).float().unsqueeze(0).to(device)

            if gameover:
                print('!GAME OVER!')
                p1_total_score.clear()
                p2_total_score.clear()
                next_state = None
            else:
                next_state = current_screen - last_screen
            try:
                player1_memory.push(state.to('cpu'),
                                    p1_action.to('cpu'),
                                    next_state.to('cpu'),
                                    torch.tensor([p1_score], device='cpu'))

                player2_memory.push(state.to('cpu'),
                                    p2_action.to('cpu'),
                                    next_state.to('cpu'),
                                    torch.tensor([p2_score], device='cpu'))
            except:
                pass

            state = next_state
            try:
                optimize(player1, player1_target, player1_optim, player1_memory)
                optimize(player2, player2_target, player2_optim, player2_memory)
            except:
                pass
            if gameover:
                break

            p1_total_score.append(p1_score)
            p2_total_score.append(p2_score)

    else:
        press(['enter', 'ctrlright'])