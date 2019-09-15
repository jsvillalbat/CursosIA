# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 20:41:45 2019

@author: sebastian villalba
"""
#Las acciones pueden ser representados en diferentes espacios, como Box, Discrete, Dict, Multi Binary etc
import gym
from gym.spaces import *
import sys
#Box => R^n (x1,x2,x3,....,xN)
#gym.spaces.Box(low = -10, high = 10, shape(2,)) #Esto genera tuplas donde el valor minimo es -10, el valor maximo es 10, y su forma son tuplas (x,y)

#Discrete => Números enteres 0 y n-1 (0,1,2,3,4,.....,n-1)
#gym.spaces.Discrete(5) #(0,1,2,3,4)

#Dict => Diccionario de espacios mas complejos
#gym.spaces.Dict({
#        "position": gym.spaces.Discrete(3), #(0,1,2)
#        "velocity": gym.spaces.Discrete(2)  #(0,1)
#        })

#Multi Binary => (0,1)*n (x1,x2,x3...,xN) Donde cada xi(0,1)
#gym.spaces.MultiBinary(3) #(x,y,z) x,y.z = T|F

#gym.spaces.MultiDiscrete([-10,10],[0,1])

#Tuple => Producto Cartesiano de espacios simples, combinaciones de diferentes espacios
#gym.spaces.Tuple(gym.spaces.Discrete(3), gym.spaces.Discrete(2)) #Tenemos (0,1,2) and (0,1) La tupla es la combinacionde estos es decir (00,01,02,10,11,12)

# =============================================================================
# Método que verifica si el espacio dado por párametro es tipo Box
#Imprime su cota superior y su cota inferior
# =============================================================================
def print_spaces(space):
    print(space)
    if isinstance(space,Box): #Comprueba si el space dado por párametro es de tipo Box de la clase Bopx
        print("\n Cota Inferior: ",space.low)
        print("\n Cota Superior: ",space.high)
        
        

if __name__ =="__main__":
    environment = gym.make(sys.argv[1]) #El usuario debe llamar el script con el nombre del videojuego como párametro
    print("Espacio de estados: ")
    print_spaces(environment.observation_space)
    print("El espacio de acciones: ")
    print_spaces(environment.action_space)
    try:
        print("Descripción de las acciones: ", environment.unwrapped.get_action_meanings())
    except AttributeError:
        pass #Pálabra reservada en Python para olvidar lo ultimo ejecutado