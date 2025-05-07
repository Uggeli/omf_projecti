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
        x = torch.flatten(x)
        x = self.output_layer(x)

        return x

def capture_screen():
    win = getWindowsWithTitle('dosbox 0.7')[0]
    try:
        win.activate()
    except:
        pass
    img = np.array(pyscreeze._screenshot_win32(region=win.box))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def tmp_matching(cur_screen, template, threshold=0.8):
    res = cv2.matchTemplate(cur_screen, template, cv2.TM_CCOEFF_NORMED)
    if res >= threshold:
        return True
    return False

def screen_preprocess(img):
    img = cv2.resize(img, (220, 320))
    return img

moves_list = [
    'up',
    'down',
    'left',
    'right',
    'down',
    'enter',
    'shiftright'
]

img_tmps = ['fight_1.png',
            'fight_2.png',
            'fight_3.png',
            'fight_4.png',
            'fight_5.png',
            '2P_fighter_select.png',
            '2Player_menu.png',
            'mainmenu.png',
            'you_win.png',
            'pre_fight_screen.png']

img_tmps = {img[:-4]: cv2.imread(img, 0) for img in img_tmps}

def take_screenshot(img):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    cv2.imwrite(fr'./training_data/{timestr}.png', img)

def save_model():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    torch.save(critic.state_dict(), fr'./critic_models/{timestr}critic.tar')

def display_game_screen(cur_screen):
    cv2.imshow('OMF', cur_screen)
    cv2.waitKey(1)

def calculate_loss(score, target_score):
    return F.smooth_l1_loss(score, torch.tensor(target_score).unsqueeze(0).to(device))

def update_model_parameters(loss):
    critic_optim.zero_grad()
    loss.backward()
    critic_optim.step()

def optimize(target_score, cur_screen):
    try:
        img_stack_tensor = torch.tensor(img_stack).float().unsqueeze(0).to(device)
        score = critic(img_stack_tensor)
        display_game_screen(cur_screen)
        if tmp_matching(cur_screen, img_tmps['you_win'], threshold=0.99999):
            target_score = 500.
        print(f'score:{score[0]:.2f}, target score:{target_score:.2f}')
        loss = calculate_loss(score, target_score)
        update_model_parameters(loss)
    except Exception as e:
        print(f"Error during optimization: {e}")

def stack_frames():
    for _ in range(4):
        cur_screen = capture_screen()
        img_stack.append(screen_preprocess(cur_screen))

device = 'cuda'
critic = Critic().to(device)

critic.load_state_dict(torch.load(r'./fighter_models/20191001-141044Fighter.tar'))
critic_optim = optim.Adagrad(critic.parameters(), lr=0.009)

img_stack = deque([screen_preprocess(capture_screen()) for _ in range(4)],
                  maxlen=4)

training = True
while training:
    cur_screen = capture_screen()
    last_state = ''
    for state, tmp in img_tmps.items():
        if 'fight_' in state and tmp_matching(cur_screen, tmp):
            fight = True
            while fight:
                cur_screen = capture_screen()
                if not tmp_matching(cur_screen, tmp):
                    fight = False
                stack_frames()
                img_stack_tensor = torch.tensor(img_stack).float().unsqueeze(0).to(device)
                score = critic(img_stack_tensor)
                print(f'Score:{score[0]:.2f}')
