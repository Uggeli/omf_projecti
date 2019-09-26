from pyautogui import getWindowsWithTitle, screenshot, press, keyDown, keyUp
import pyscreeze
import cv2
import numpy as np

from collections import deque

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

        self.output_layer = nn.Sequential(
            nn.Linear(25760, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        x = self.conv_layer(x)
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


def tmp_matching(cur_screen, template):
    res = cv2.matchTemplate(cur_screen, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    if res >= threshold:
        return True
    return False


def screen_preprocess(img):
    # img = transforms.functional.resize(img, 220, 320)
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
            'you_win.png']

img_tmps = {img[:-4]: cv2.imread(img, 0) for img in img_tmps}


def take_screenshot(img):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    cv2.imwrite(fr'G:\Kdaus\pyyttoni\Omf_projekti2\training_data\{timestr}.png',
                img)


device = 'cuda'
critic = Critic().to(device)
critic_optim = optim.Adagrad(critic.parameters())

img_tmps = {img[:-4]: cv2.imread(img, 0) for img in img_tmps}

img_stack = deque([screen_preprocess(capture_screen()) for _ in range(4)],
                  maxlen=4)

cur_screen = capture_screen()
while tmp_matching(cur_screen, img_tmps['fight_1']):
    keyDown('right')
    cur_screen = capture_screen()
    # take_screenshot(cur_screen)
    time.sleep(1)
    keyUp('right')
    press('enter')
    time.sleep(0.18)
    cur_screen = capture_screen()
    # take_screenshot(cur_screen)
    keyDown('left')
    cur_screen = capture_screen()
    # take_screenshot(cur_screen)
    time.sleep(1)
    keyUp('left')
    press('enter')
    time.sleep(0.18)
    cur_screen = capture_screen()
    # take_screenshot(cur_screen)
