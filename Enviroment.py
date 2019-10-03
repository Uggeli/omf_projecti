from pyautogui import getWindowsWithTitle, screenshot, press, keyDown, keyUp
import pyscreeze
import cv2
import numpy as np
import os


class Enviroment:
    def __init__(self):
        def _create_fight_tmps():
            img_tmps = [
                'fight_1.png',
                'fight_2.png',
                'fight_3.png',
                'fight_4.png',
                'fight_5.png']
            return [cv2.imread(img, 0) for img in img_tmps]

        def _create_menu_tmps():
            img_tmps = [
                '2P_fighter_select.png',
                '2Player_menu.png',
                'mainmenu.png',
                'you_win.png',
                'pre_fight_screen.png']
            img_tmps = {img[:-4]: cv2.imread(img, 0) for img in img_tmps}
            return img_tmps

        self.tmps = {}
        self.tmps['fight'] = _create_fight_tmps()
        self.tmps['menu'] = _create_menu_tmps()
        self.win = getWindowsWithTitle('dosbox 0.7')[0]
        self.last_state = ''

    def capture_screen(self):
        try:
            self.win.activate()
        except:
            pass
        img = np.array(pyscreeze._screenshot_win32(region=self.win.box))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def tmp_matching(self, cur_screen, template, threshold=0.8):
        res = cv2.matchTemplate(cur_screen, template, cv2.TM_CCOEFF_NORMED)
        # threshold = 0.78
        if res >= threshold:
            return True
        return False

    def screen_preprocess(img):
        # img = transforms.functional.resize(img, 220, 320)
        img = cv2.resize(img, (220, 320))
        return img

    def match(self):
        img = self.capture_screen()
        for name, temp in self.tmps.items():
            if name == 'fight':
                for tmp in temp:
                    if self.tmp_matching(img, tmp):
                        self.last_state = name
                        return name

            else:
                for menu_item, m_tmp in temp.items():
                    if self.tmp_matching(img, m_tmp):
                        self.last_state = menu_item
                        return menu_item

    def get_fight_status(self):
        img = self.capture_screen()

        # cropped = img[0:72, 0:640]
        p1_health_bar = img[50:60, 0:216]
        p2_health_bar = img[50:60, 434:640]
        # p1_health = np.int0(cv2.goodFeaturesToTrack(
        #                 p1_health_bar, 100, 0.5, 10))

        # p2_health = np.int0(cv2.goodFeaturesToTrack(
        #                 p2_health_bar, 100, 0.5, 10))

        # for p1, p2 in zip(p1_health, p2_health):
        #     x_p1, y_p1 = p1.ravel()
        #     cv2.circle(p1_health_bar, (x_p1, y_p1), 3, 255, -1)
        #     x_p2, y_p2 = p2.ravel()
        #     cv2.circle(p2_health_bar, (x_p2, y_p2), 3, 255, -1)

        p1_health = cv2.goodFeaturesToTrack(
                        p1_health_bar, 100, 0.5, 10)

        p2_health = cv2.goodFeaturesToTrack(
                        p2_health_bar, 100, 0.5, 10)
        try:
            p1_health_measure = p1_health.ravel()
            p1_current_health = p1_health_measure[0] - p1_health_measure[-2]

            p2_health_measure = p2_health.ravel()
            p2_current_health = p2_health_measure[0] - p2_health_measure[-2]

            # print(f'P1:{p1_current_health} - {p2_current_health}:P2')
            # cv2.imshow('p1', p1_health_bar)
            # cv2.imshow('p2', p2_health_bar)

            # cv2.waitKey(1)

            return p1_current_health, p2_current_health
        except:
            return 0, 0

env = Enviroment()

while True:
    print(env.match())
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
            # print(env.get_fight_status())
            p1, p2 = env.get_fight_status()
            if p1 >= 200:
                p2_score = 100
                p1 = p1_last
                gameover = True
            else:
                p2_score = p1_last - p1
                p1_last = p1
                # gameover = False

            # print(p1)

            if p2 >= 200:
                p1_score = 100
                p2 = p2_last
                gameover = True
            else:
                p1_score = p2_last - p2
                p2_last = p2
                # gameover = False
            
            p1_total_score.append(p1_score)
            p2_total_score.append(p2_score)
            # print(p2)

            # print(f'P1:{p1} - {p2}:P2')
            # print('========================')
            # os.system('cls')
            print(f'P1:\t\t\tP2:\
                \nHealth:{p1}\t\tHealth:{p2}\
                \nScore:{sum(p1_total_score)}\t\tScore:{sum(p2_total_score)}')
