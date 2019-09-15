# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 23:00:25 2019

@author: sebastian villalba
"""
from collections import namedtuple
import random

Experience = namedtuple("Experience",["obs","action","reward","next_obs","done"])

class ExperienceMemory(object):
    #1)Se debe guardar nuevas experiencias dentro de la estructura de datos, basado en los datos que se recoelctan del entorno
    #2)Se debe recuperar fragmentos de experiencias para rejugar la funcion Q de forma mas rápida, si ya se sabe pasar una forma se consulta la Experience
    #Esto es un buffer que simula la memoria de experiencia del agente
    def __init__(self,capacity = int(1e6), ):
        """
        :param capacity: Capacidad de la memoria cíclica, numero maximo de experiencias almacenables( se va desechando la información más vieja)
        :return:
        """
        self.capacity = capacity
        self.memory_idx = 0 #identificador que sabe la experiencia actual
        self.memory = []
        
    def sample(self, batch_size):
        """
        :param batch_size: tamaño de la memoria a recuperar
        :return: Una lista del tamaño batch_size de experiencias aleatorias de la memoria(Una muestra aleatoria de la memoria)
        """
        assert batch_size <= self.get_size(),"El tamaño de la muestra es superior a la memoria disponible"
        return random.sample(self.memory,batch_size)
    
    def get_size(self):
        """
        :return_ numero de experiencias almacenadas en memoria
        """
        return len(self.memory)
    
    def store(self, exp):
        """
        :param exp: Objeto experiencia a ser alamcenada en memoria
        """
        self.memory.insert(self.memory_idx % self.capacity,exp) #Para que sea ciclico se coloca el módulo
        self.memory_idx += 1
        
        
    