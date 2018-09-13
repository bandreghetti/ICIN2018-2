#!/usr/bin/env python3

import gym
import time
import pynput.keyboard as kb
from joypad import Joypad
from games import gameList
from agents import MultilayerPerceptron

FPS = 30.0
framePeriod = 1.0/30.0

joypad = Joypad()

def main():
    kbListener = kb.Listener(on_press=joypad.on_press, on_release=joypad.on_release)
    kbListener.start()

    env = gym.make(gameList[1])
    agent = MultilayerPerceptron(env.observation_space)
    
    env.reset()
    observation, _, done, _ = env.step(env.action_space.sample())
    env.render()
    lastRender = time.time()
    try:
        while(True):
            action = joypad.action
            if action == -1:
                break
            if time.time() - lastRender > framePeriod:
                lastRender = time.time()
                observation, _, done, _ = env.step(action)
                agent.train(observation, joypad.actionButtons())
                print('{} {} Autoplay:{}'.format(joypad.actionButtons(),
                                                 joypad.lastKey,
                                                 joypad.autoplay))
                print(agent.action(observation))
                if done:
                    env.reset()
                env.render()
    except KeyboardInterrupt:
        pass
    env.close()
    kbListener.stop()

main()