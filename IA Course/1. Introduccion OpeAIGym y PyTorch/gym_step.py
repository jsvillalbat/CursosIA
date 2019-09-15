# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 22:06:13 2019

@author: sebastian villalba
"""

import gym as gm 

enviroment = gm.make("Qbert-v0") # Lanzamos o creamos un ambiente de la montaña Rusa
MAX_NUM_EPISODES = 10
MAX_STEPS_PER_EPISODE = 500

for episode in range(MAX_NUM_EPISODES): # Durante 2000 iteraciones
  obs = enviroment.reset() # Observacion
  for step in range(MAX_STEPS_PER_EPISODE):
      enviroment.render()
      action = enviroment.action_space.sample() #$Tomamos una decision aleatoria
      next_state,reward,done,inf = enviroment.step(action)
      obs = next_state
      
      if done is True:
          print("\n Episodio #{} terminado en {} steps.".format(episode,step+1))
          break
enviroment.close() #Cierra el ambiente creada o la sesion de OpenAiGym, después de las 2000 iteraciones
