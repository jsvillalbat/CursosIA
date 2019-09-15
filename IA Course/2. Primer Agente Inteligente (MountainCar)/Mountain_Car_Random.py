# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 22:30:14 2019
@author: sebastian villalba
"""
#El problema Entorno Mountain Car
#Espacio de observaciones:
#Box(2,)
#
# Cota Inferior:  [-1.2  -0.07] la primera tupla representa posicion va desde (-1.2,0.6), segunda velocidad desde (-0.07,0.07)
#
# Cota Superior:  [0.6  0.07]
#El espacio de acciones:
#Discrete(3) #acciones, (0,1,2) 0 = izquierda, 1 = no hacer nada, 2 = derecha

import gym
environment = gym.make("MountainCar-v0")
MAX_NUM_EPISODES = 1000

for episode in range(MAX_NUM_EPISODES):
    done = False
    obs = environment.reset()
    total_reward = 0.0 # Variable para guardar la recompensa total en cada episodio
    step = 0
    while not done:
        environment.render()
        action = environment.action_space.sample() # Accion aleatorio que posteriormente se cabmiara por la decision de nuestro ageten inteligente
        next_state,reward,done,info = environment.step(action)
        total_reward += reward
        step += 1
        obs = next_state
        
    print("\n Episodio número {} finalizado con {} iteraciones. Recompensa final = {}".format(episode,step+1,total_reward))
    
environment.close()



    
    
        
    
