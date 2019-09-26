from pyautogui import getWindowsWithTitle, screenshot, press, keyDown, keyUp
import pyscreeze
import cv2
# from PIL import ImageGrab
import numpy as np
import time

from collections import deque, namedtuple

import torch
import torch.nn as nn
# from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F

import random

Memory = namedtuple('Memory',
                    ('last_state','state', 'action', 'reward'))


class Fighter(nn.Module):
    def __init__(self):
        super(Fighter, self).__init__()

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
            nn.Linear(hidden, 6)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        # x = torch.flatten(x)
        x = self.output_layer(x.view(x.size(0), -1))

        return x


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


def capture_screen():
    win = getWindowsWithTitle('dosbox 0.7')[0]
    # try:
    #     win.activate()
    # except:
    #     pass
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


device = 'cuda'

replay_memory = Replay_memory(1_000_000)

brain = Fighter().to(device)
critic = Critic().to(device)
critic_optim = optim.Adagrad(critic.parameters())

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
            'mainmenu.png']

img_tmps = {img[:-4]: cv2.imread(img, 0) for img in img_tmps}

img_stack = deque([screen_preprocess(capture_screen()) for _ in range(4)],
                  maxlen=4)


# while True:
#     cur_screen = capture_screen()
#     for name, tmp in img_tmps.items():
#         if tmp_matching(cur_screen, tmp):
#             if 'fight_' in name:
#                 # press(random.choice(moves_list))
#                 img_stack.append(screen_preprocess(cur_screen))
#                 img_stack_tensor = torch.tensor(img_stack).float().unsqueeze(0).to(device)
#                 action = brain(img_stack_tensor).max(1)[1].view(1, 1)
#                 press(moves_list[action.item()])



#                 # print(f'fight!')
#             else:
#                 print('not in fight')

def send_action(action):
    win = getWindowsWithTitle('dosbox 0.7')[0]
    win.activate()
    press(action,pause=0.5)
    time.sleep(0.14)
    # keyDown('p')
    # keyUp('p')


while True:
    win = getWindowsWithTitle('dosbox 0.7')[0]
    cur_screen = capture_screen()
    for name, tmp in img_tmps.items():
        if tmp_matching(cur_screen, tmp):

            if 'fight_' in name:
                # print('in fight!')
                time.sleep(0.014)
                while tmp_matching(cur_screen, tmp):
                    
                    action = random.choice(moves_list)
                    send_action(action)
                    time.sleep(0.14)
                    cur_screen = capture_screen()
                    img_stack.append(screen_preprocess(cur_screen))
                    img_stack_tensor = torch.tensor(img_stack).float().unsqueeze(0).to(device)
                    score = critic(img_stack_tensor)

                    print(f'nn antoi pisteet: {score[0]} edellinen liike:{action}')
                    # target_score = float(input('Pisteet: '))
                    if action == 'enter' or action == 'shiftright':
                        target_score = 10.
                    else:
                        target_score = .0

                    win.activate()
                    cv2.imshow('Preview', cur_screen)
                    cv2.waitKey(1)
                    # keyDown('p')
                    # keyUp('p')

                    # optim critic
                    loss = F.smooth_l1_loss(score, torch.tensor(target_score).unsqueeze(0).to(device))

                    critic_optim.zero_grad()
                    loss.backward()
                    for param in critic.parameters():
                        param.grad.data.clamp_(-1, 1)
                    critic_optim.step()
                
                timestr = time.strftime("%Y%m%d-%H%M%S")
                torch.save(critic.state_dict(), f'{timestr}critic.tar')





            # if 'menu' in name:
            #     print('in main menu')
            #     press('down', pause=1)
            #     press('enter', pause=1)

            # if 'fighter_select' in name:
            #     print('fighter select')
            #     press('enter')
