from pyautogui import getWindowsWithTitle, screenshot, press, keyDown, keyUp
import pyscreeze
import cv2
import numpy as np

from collections import deque

import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Enviroment2 import Enviroment


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

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
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.conv_layer2(x)
        # x = self.conv_layer3(x)
        x = torch.flatten(x)
        x = self.output_layer(x)

        return x

moves_list = [
    'up',
    'down',
    'left',
    'right',
    'down',
    'enter',
    'shiftright'
]


def save_model():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    torch.save(critic.state_dict(),
               fr'G:\Kdaus\pyyttoni\Omf_projekti2\critic_models\{timestr}critic.tar')


def optimize(target_score, cur_screen):
    img_stack_tensor = torch.tensor(img_stack).float().unsqueeze(0).to(device)
    score = critic(img_stack_tensor)
    cv2.imshow('OMF', cur_screen)
    cv2.waitKey(1)    
    print(f'score:{score[0]:.2f}, target score:{target_score:.2f}')
    loss = F.smooth_l1_loss(score, torch.tensor(
        target_score).unsqueeze(0).to(device))
    # critic.zero_grad()
    critic_optim.zero_grad()
    loss.backward()
    # for param in critic.parameters():
    #     param.grad.data.clamp_(-1, 1)
    critic_optim.step()


def stack_frames():
    for _ in range(4):
        cur_screen = capture_screen()
        img_stack.append(screen_preprocess(cur_screen))

env = Enviroment()

device = 'cuda'
critic = Critic().to(device)

critic.load_state_dict(torch.load(
    r'G:\Kdaus\pyyttoni\Omf_projekti2\critic_models\20190928-204735critic.tar'))

critic_optim = optim.Adagrad(critic.parameters(), lr=0.009)

img_stack = deque([env.screen_preprocess(env.capture_screen()) for
                  _ in range(4)],
                  maxlen=4)

training = True
while training:
    if env.match() == 'fight':
        in_fight = True
        gameover = False

        p1_last = 196
        p2_last = 196

        p1_total_score = []
        p2_total_score = []

        while in_fight:
            if env.match() != 'fight':
                in_fight = False
                break

            if gameover:
                print('!GAME OVER!')
                p1_total_score.clear()
                p2_total_score.clear()
                break

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
            # if not gameover and env.match() == 'fight':
            try:
                img_stack.append(env.screen_preprocess(env.capture_screen()))
                optimize(p2_score, env.capture_screen())
            except:
                pass

            p1_total_score.append(p1_score)
            p2_total_score.append(p2_score)

            # print(f'P1:\t\t\tP2:\
            #     \nHealth:{p1}\t\tHealth:{p2}\
            #     \nScore:{sum(p1_total_score)}\t\
            #     Score:{sum(p2_total_score)}')
