from pynput.keyboard import Key
import numpy as np

actionMeaning = {
    0 : "NOOP",
    1 : "FIRE",
    2 : "UP",
    3 : "RIGHT",
    4 : "LEFT",
    5 : "DOWN",
    6 : "UPRIGHT",
    7 : "UPLEFT",
    8 : "DOWNRIGHT",
    9 : "DOWNLEFT",
    10 : "UPFIRE",
    11 : "RIGHTFIRE",
    12 : "LEFTFIRE",
    13 : "DOWNFIRE",
    14 : "UPRIGHTFIRE",
    15 : "UPLEFTFIRE",
    16 : "DOWNRIGHTFIRE",
    17 : "DOWNLEFTFIRE",
}

NOOP          = 0
FIRE          = 1
UP            = 2
RIGHT         = 3
LEFT          = 4
DOWN          = 5
UPRIGHT       = 6
UPLEFT        = 7
DOWNRIGHT     = 8
DOWNLEFT      = 9
UPFIRE        = 10
RIGHTFIRE     = 11
LEFTFIRE      = 12
DOWNFIRE      = 13
UPRIGHTFIRE   = 14
UPLEFTFIRE    = 15
DOWNRIGHTFIRE = 16
DOWNLEFTFIRE  = 17

class Joypad():
    def __init__(self):
        self.action = NOOP
        self.lastKey = None
    
    def actionString(self):
        return actionMeaning[self.action]

    def actionOneHot(self):
        ret = np.zeros(18, dtype=np.float64)
        np.put(ret, self.action, 1)
        return ret

    def on_press(self, key):
        self.lastKey = key
        if(key == Key.space): 
            if  (self.action == NOOP):
                self.action = FIRE
            elif(self.action == UP):
                self.action = UPFIRE
            elif(self.action == RIGHT):
                self.action = RIGHTFIRE
            elif(self.action == LEFT):
                self.action = LEFTFIRE
            elif(self.action == DOWN):
                self.action = DOWNFIRE
            elif(self.action == UPRIGHT):
                self.action = UPRIGHTFIRE
            elif(self.action == UPLEFT):
                self.action = UPLEFTFIRE
            elif(self.action == DOWNRIGHT):
                self.action = DOWNRIGHTFIRE
            elif(self.action == DOWNLEFT):
                self.action = DOWNLEFTFIRE
        elif(key == Key.up):
            if  (self.action == NOOP):
                self.action = UP
            elif(self.action == FIRE):
                self.action = UPFIRE
            elif(self.action == RIGHT):
                self.action = UPRIGHT
            elif(self.action == LEFT):
                self.action = UPLEFT
            elif(self.action == RIGHTFIRE):
                self.action = UPRIGHTFIRE
            elif(self.action == LEFTFIRE):
                self.action = UPLEFTFIRE
        elif(key == Key.right):
            if  (self.action == NOOP):
                self.action = RIGHT
            elif(self.action == FIRE):
                self.action = RIGHTFIRE
            elif(self.action == UP):
                self.action = UPRIGHT
            elif(self.action == DOWN):
                self.action = DOWNRIGHT
            elif(self.action == UPFIRE):
                self.action = UPRIGHTFIRE
            elif(self.action == DOWNFIRE):
                self.action = DOWNRIGHTFIRE
        elif(key == Key.left):
            if  (self.action == NOOP):
                self.action = LEFT
            elif(self.action == FIRE):
                self.action = LEFTFIRE
            elif(self.action == UP):
                self.action = UPLEFT
            elif(self.action == DOWN):
                self.action = DOWNLEFT
            elif(self.action == UPFIRE):
                self.action = UPLEFTFIRE
            elif(self.action == DOWNFIRE):
                self.action = DOWNLEFTFIRE
        elif(key == Key.down):
            if  (self.action == NOOP):
                self.action = DOWN
            elif(self.action == FIRE):
                self.action = DOWNFIRE
            elif(self.action == RIGHT):
                self.action = DOWNRIGHT
            elif(self.action == LEFT):
                self.action = DOWNLEFT
            elif(self.action == RIGHTFIRE):
                self.action = DOWNRIGHTFIRE
            elif(self.action == LEFTFIRE):
                self.action = DOWNLEFTFIRE

    def on_release(self, key):
        if(key == Key.space):
            if  (self.action == FIRE):
                self.action = NOOP
            elif(self.action == UPFIRE):
                self.action = UP
            elif(self.action == RIGHTFIRE):
                self.action = RIGHT
            elif(self.action == LEFTFIRE):
                self.action = LEFT
            elif(self.action == DOWNFIRE):
                self.action = DOWN
            elif(self.action == UPRIGHTFIRE):
                self.action = UPRIGHT
            elif(self.action == UPLEFTFIRE):
                self.action = UPLEFT
            elif(self.action == DOWNRIGHTFIRE):
                self.action = DOWNRIGHT
            elif(self.action == DOWNLEFTFIRE):
                self.action = DOWNLEFT
        elif(key == Key.up):
            if  (self.action == UP):
                self.action = NOOP
            elif(self.action == UPFIRE):
                self.action = FIRE
            elif(self.action == UPRIGHT):
                self.action = RIGHT
            elif(self.action == UPLEFT):
                self.action = LEFT
            elif(self.action == UPRIGHTFIRE):
                self.action = RIGHTFIRE
            elif(self.action == UPLEFTFIRE):
                self.action = LEFTFIRE
        elif(key == Key.right):
            if  (self.action == RIGHT):
                self.action = NOOP
            elif(self.action == RIGHTFIRE):
                self.action = FIRE
            elif(self.action == UPRIGHT):
                self.action = UP
            elif(self.action == DOWNRIGHT):
                self.action = DOWN
            elif(self.action == UPRIGHTFIRE):
                self.action = UPFIRE
            elif(self.action == DOWNRIGHTFIRE):
                self.action = DOWNFIRE
        elif(key == Key.left):
            if  (self.action == LEFT):
                self.action = NOOP
            elif(self.action == LEFTFIRE):
                self.action = FIRE
            elif(self.action == UPLEFT):
                self.action = UP
            elif(self.action == DOWNLEFT):
                self.action = DOWN
            elif(self.action == UPLEFTFIRE):
                self.action = UPFIRE
            elif(self.action == DOWNLEFTFIRE):
                self.action = DOWNFIRE
        elif(key == Key.down):
            if  (self.action == DOWN):
                self.action = NOOP
            elif(self.action == DOWNFIRE):
                self.action = FIRE
            elif(self.action == DOWNRIGHT):
                self.action = RIGHT
            elif(self.action == DOWNLEFT):
                self.action = LEFT
            elif(self.action == DOWNRIGHTFIRE):
                self.action = RIGHTFIRE
            elif(self.action == DOWNLEFTFIRE):
                self.action = LEFTFIRE

