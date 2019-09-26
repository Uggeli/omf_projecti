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


def tmp_matching(cur_screen, template, threshold=0.8):
    res = cv2.matchTemplate(cur_screen, template, cv2.TM_CCOEFF_NORMED)
    # threshold = 0.78
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
            'you_win.png',
            'pre_fight_screen.png']

img_tmps = {img[:-4]: cv2.imread(img, 0) for img in img_tmps}


def take_screenshot(img):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    cv2.imwrite(fr'G:\Kdaus\pyyttoni\Omf_projekti2\training_data\{timestr}.png', img)


def save_model():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    torch.save(critic.state_dict(), fr'G:\Kdaus\pyyttoni\Omf_projekti2\critic_models\{timestr}critic.tar')


def optimize(target_score, cur_screen):
    img_stack_tensor = torch.tensor(img_stack).float().unsqueeze(0).to(device)
    score = critic(img_stack_tensor)
    cv2.imshow('OMF', cur_screen)
    cv2.waitKey(1)
    if tmp_matching(cur_screen, img_tmps['you_win'], threshold=0.99999):
        target_score = 500.
    print(f'score:{score[0]:.2f}, target score:{target_score:.2f}')
    loss = F.smooth_l1_loss(score, torch.tensor(target_score).unsqueeze(0).to(device))
    critic_optim.zero_grad()
    loss.backward()
    for param in critic.parameters():
        param.grad.data.clamp_(-1, 1)
    critic_optim.step()


def stack_frames():
    for _ in range(4):
        cur_screen = capture_screen()
        img_stack.append(screen_preprocess(cur_screen))

device = 'cuda'
critic = Critic().to(device)
critic_optim = optim.Adagrad(critic.parameters())

img_stack = deque([screen_preprocess(capture_screen()) for _ in range(4)],
                  maxlen=4)

training = True
while training:
    cur_screen = capture_screen()
    last_state = ''
    while tmp_matching(cur_screen, img_tmps['fight_1']):

        last_state = 'fight'

        keyDown('right')
        stack_frames()
        cur_screen = capture_screen()
        optimize(0., cur_screen)
        time.sleep(1)

        keyUp('right')
        stack_frames()
        cur_screen = capture_screen()
        optimize(2., cur_screen)

        press('enter')
        stack_frames()
        cur_screen = capture_screen()
        optimize(10., cur_screen)

        keyDown('left')
        stack_frames()
        cur_screen = capture_screen()
        optimize(0., cur_screen)
        time.sleep(1)

        keyUp('left')
        stack_frames()
        cur_screen = capture_screen()
        optimize(0., cur_screen)
        time.sleep(1)

        press('enter')
        stack_frames()
        cur_screen = capture_screen()
        optimize(0., cur_screen)
    
    if last_state == 'fight':
        save_model()

    if tmp_matching(cur_screen, img_tmps['2P_fighter_select']) or tmp_matching(cur_screen, img_tmps['2Player_menu']) or tmp_matching(cur_screen, img_tmps['pre_fight_screen']):
        last_state = 'select'
        press('enter')
        press('ctrlright')
