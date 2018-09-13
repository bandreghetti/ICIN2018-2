#!/usr/bin/env python3

import gym
import time
import pynput.keyboard as kb
from joypad import Joypad

joypad = Joypad()

def main():
    kbListener = kb.Listener(on_press=joypad.on_press, on_release=joypad.on_release)
    kbListener.start()
    
    env = gym.make('Riverraid-v0')
    env.reset()
    for _ in range(2400):
        env.render()
        # action = env.action_space.sample()
        print('{} {} {}'.format(joypad.actionString(), joypad.action, joypad.lastKey))
        _, _, done, _ = env.step(joypad.action) # take a random action
        time.sleep(1.0/30.0)
        if done:
            env.reset()
            
    env.close()
    kbListener.stop()

main()