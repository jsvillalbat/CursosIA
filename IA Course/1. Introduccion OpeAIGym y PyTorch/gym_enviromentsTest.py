# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 19:59:56 2019

@author: sebastian villalba
"""
from gym import envs # Solo importe el paquete de ambientes de la libreria gym

env_names = {env.id for env in envs.registry.all()} # Dame los nombres o id de cada ambiente en todo  el registro de ambientes

for name in sorted(env_names): #imprime cada nombre en la lista ordenada alfabeticamente de env_names
    print(name)