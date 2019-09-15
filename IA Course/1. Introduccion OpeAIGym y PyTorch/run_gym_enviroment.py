# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 20:26:48 2019

@author: sebastian villalba
"""

import gym
import sys 

def run_gym_enviroment(argv):
# El primer párametro de argu será el nombre del entorno a ejecutar
    enviroment = gym.make(argv[1])
    enviroment.reset() # inicializar el entorno
    for _ in range(int(argv[2])):
            enviroment.render()
            enviroment.step(enviroment.action_space.sample()) # Accion aleatoria
    enviroment.close()
    
if __name__ == "__main__":
    run_gym_enviroment(sys.argv)