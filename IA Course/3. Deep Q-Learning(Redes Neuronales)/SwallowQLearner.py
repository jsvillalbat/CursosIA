## -*- coding: utf-8 -*-
#"""
#Created on Sat Jun 15 15:43:21 2019
#
#@author: sebastian villalba
#"""
#
#import numpy as np
#import torch
#from libs.perceptron import SLP
#from utils.decay_schedule import LinearDecaySchedule
#import random
#import gym
#from utils.experience_memory import Experience_memory,Experience
##QLearner Class
##__init(self,environment)
##discretize(self,obs)
##get_action(self,obs)
##learn(self,obs,action,reward,next_obs)
##
##EPSILON_MIN: vamos aprendiendo, mientras el incremento de aprendizaje sea superior a dicho valot
##MAX_NUM_EPISODES: número máximo de iteraciones que estamos dispuestos a realizar
##STEPS_PER_EPISODE: número máximo de pasos a realizar en cada episodio
##ALPHA: ratio de aprendizaje del agente
##GAMMA: factor de descuento del agente
##NUM_DISCRETE_BINS: número de divisiones en el caso de discretizar el espacio de estados continuo.
#
#MAX_NUM_EPISODES = 100000
#STEPS_PER_EPISODE = 300 #Este valor es propio del videojuego el Mountain Car contiene 200 episodios para pasarse el juego
#
#
#class SwallowQLearner(object):
#    
#    #Metodo para inicializar las variables, es como el metodo constructor
#    def __init__(self,environment,learning_rate = 1e-5,gamma = 0.98):
#        self.obs_shape = environment.observation_space.shape        
#        self.action_shape = environment.action_space.n
#        self.Q = SLP(self.obs_shape,self.action_shape) #Nuestro Q es un perceptron, que recibe todo el espacio de observaciones y el valor de salida es todo el espacio de las acciones
#        self.Q_optimizer = torch.optim.Adam(self.Q.parameters() , lr = learning_rate) # Ratio de aprendizaje, que porcentaje vamos aprendiendo de un paso a otro y utiliza el algoritmo de Adam, el método parametros es de Neural Netowork Module
#        self.gamma = gamma
#        self.epsilon_max = 1.0
#        self.epsilon_min = 0.05
#        self.epsion_decay = LinearDecaySchedule(initial_value = self.epsilon_max, 
#                                                final_value = self.epsilon_min,
#                                                max_steps = 0.5*MAX_NUM_EPISODES*STEPS_PER_EPISODE)
#        self.step_num = 0
#        self.policy = self.epsilon_greedy_Q
#        self.memory = Experience_memory(capacity= int(1e5))
#        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#        
#    #Método para elegir una accion dada una observacion
#    def get_action(self,obs):
#        return self.policy(obs)
#    
#    
#    def epsilon_greedy_Q(self,obs):
#        if random.random() < self.epsion_decay(self.step_num):
#            action = random.choice([a for a in range(self.action_shape) ])
#        else:
#            action =  np.argmax(self.Q(obs).data.to(torch.device('cpu')).numpy())
#        return action
#  
#    #Implementación del aprendizaje de red neuronal    
#    def learn(self,obs,action,reward,next_obs):
#        td_target = reward + self.gamma * torch.max(self.Q(next_obs))
#        td_error = torch.nn.functional.mse_loss(self.Q(obs)[action],td_target) #Mean square error, eror cuadratico medio mide el error
#        self.Q_optimizer.zero_grad()
#        td_error.backward() #Hace el procesos hacia atras, que es lo que le ha llevado a esa decision, backpropagation
#        self.Q_optimizer.step()#Optimiza los valores o pesos internos de la red neuronal
# 
#    def replay_experience(self, batch_size):
#        """
#        Vuelve a jugar usando la experiencia aleatoria almacenada
#        :param batch_size: Tamaño de la muestra a tomar de la memoria
#        :return:
#        """
#        experience_batch = self.memory.sample(batch_size)
#        self.learn_from_batch_experience(experience_batch)   
#        
#    
#        
#    def learn_from_batch_experience(self, experiences):
#       """
#       Actualiza la red neuronal profunda en base a lo aprendido en el conjunto de experiencias anteriores
#       :param experiences: Fragmento de recuerdos anteriores
#       :return:
#       """
#       batch_xp = Experience(*zip(*experiences)) # el asterisco indica que le pasa una referencia, apuntador
#       obs_batch = np.array(batch_xp.obs)
#       action_batch = np.array(batch_xp.action)
#       reward_batch = np.array(batch_xp.reward)
#       next_obs_batch = np.array(batch_xp.next_obs)
#       done_batch = np.array(batch_xp.done)
#       
#       td_target = reward_batch + ~done_batch * \
#               np.tile(self.gamma, len(next_obs_batch))*\
#               self.Q(next_obs_batch).detach().max(1)[0].data #Madan error al compilar
#               
#       td_target = td_target.to(self.device)
#       action_idx = torch.from_numpy(action_batch).to(self.device)
#       td_error = torch.nn.functional.mse_loss(self.Q(obs_batch).gather(1, action_idx.view(-1,1)),
#                                               td_target.float().unsqueeze(1))
#       self.Q_optimizer.zero_grad()
#       td_error.mean().backward()
#       self.Q_optimizer.step()
#       
#       
#       
#if __name__ == "__main__":
#    environment = gym.make("CartPole-v0")
#    agent = SwallowQLearner(environment)
#    first_episode = True
#    episode_rewards = list()
#    for episode in range(MAX_NUM_EPISODES):
#        obs = environment.reset()
#        total_reward = 0
#        for step in range(STEPS_PER_EPISODE):
#            #environment.render()
#            action = agent.get_action(obs)
#            next_obs ,reward,done,info = environment.step(action)
#            agent.memory.store(Experience(obs,action,reward, next_obs,done))
#            agent.learn(obs, action, reward,next_obs)
#            
#            obs = next_obs
#            total_reward+= reward
#            if done is True:
#                if first_episode:
#                    max_reward = total_reward
#                    first_episode = False
#                episode_rewards.append(total_reward)
#                if total_reward > max_reward:
#                    max_reward = total_reward
#                print("\nEpisodio {} finalizado con {} iteraciones. Recompensa = {}, Recompensa media = {}, Mejor Recompensa = {}".format(episode,step+1, total_reward, np.mean(episode_rewards), max_reward))
#                if agent.memory.get_size()>100:
#                    agent.replay_experience(32) # En base a las ultimas 32 experiencias prubeas ir mejorando nuestro agente
#                break
#    environment.close()
#                    

    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 11:13:57 2018
@author: juangabriel
"""

import torch
import numpy as np
from libs.perceptron import SLP
from utils.decay_schedule import LinearDecaySchedule
import random
import gym
from utils.experience_memory import ExperienceMemory, Experience

MAX_NUM_EPISODES = 100000
STEPS_PER_EPISODE = 300


class SwallowQLearner(object):
    def __init__(self, environment, learning_rate = 0.005, gamma = 0.98):
        self.obs_shape = environment.observation_space.shape
        
        self.action_shape = environment.action_space.n
        self.Q = SLP(self.obs_shape, self.action_shape)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr = learning_rate)
        
        self.gamma = gamma
        
        self.epsilon_max = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = LinearDecaySchedule(initial_value = self.epsilon_max,
                                                 final_value = self.epsilon_min, 
                                                 max_steps = 0.5 * MAX_NUM_EPISODES * STEPS_PER_EPISODE)
        self.step_num = 0
        self.policy = self.epsilon_greedy_Q
        
        self.memory = ExperienceMemory(capacity = int(1e5))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
         
    def get_action(self, obs):
        return self.policy(obs)
    
    def epsilon_greedy_Q(self, obs):
        if random.random() < self.epsilon_decay(self.step_num):
            action = random.choice([a for a in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(obs).data.to(torch.device('cpu')).numpy())   
        return action
        
        
    def learn(self, obs, action, reward, next_obs):
        td_target = reward + self.gamma * torch.max(self.Q(next_obs))
        td_error = torch.nn.functional.mse_loss(self.Q(obs)[action], td_target)
        self.Q_optimizer.zero_grad()
        td_error.backward()
        self.Q_optimizer.step()
        
    def replay_experience(self, batch_size):
        """
        Vuelve a jugar usando la experiencia aleatoria almacenada
        :param batch_size: Tamaño de la muestra a tomar de la memoria
        :return: 
        """
        experience_batch = self.memory.sample(batch_size)
        self.learn_from_batch_experience(experience_batch)   
      
    def learn_from_batch_experience(self, experiences):
        """
        Actualiza la red neuronal profunda en base a lo aprendido en el conjunto de experiencias anteriores
        :param experiences: fragmento de recuerdos anteriores
        :return: 
        """
        batch_xp = Experience(*zip(*experiences))
        obs_batch = np.array(batch_xp.obs)
        action_batch = np.array(batch_xp.action)
        reward_batch = np.array(batch_xp.reward)
        next_obs_batch = np.array(batch_xp.next_obs)
        done_batch = np.array(batch_xp.done)
        
        td_target = reward_batch + ~done_batch * \
                    np.tile(self.gamma, len(next_obs_batch)) * \
                    self.Q(next_obs_batch).detach().max(1)[0].data
        
        td_target = td_target.to(self.device)
        action_idx = torch.from_numpy(action_batch).to(self.device)
        td_error = torch.nn.functional.mse_loss(
                self.Q(obs_batch).gather(1, action_idx.view(-1,1)),
                td_target.float().unsqueeze(1))
        
        self.Q_optimizer.zero_grad()
        td_error.mean().backward()
        self.Q_optimizer.step()
        
        
        
        
        
    
if __name__ == "__main__":
    environment = gym.make("CartPole-v0")
    agent = SwallowQLearner(environment)
    first_episode = True
    episode_rewards = list()
    for episode in range(MAX_NUM_EPISODES):
        obs = environment.reset()
        total_reward = 0.0
        for step in range(STEPS_PER_EPISODE):
            #environment.render()
            action = agent.get_action(obs)
            next_obs, reward, done, info = environment.step(action)
            agent.memory.store(Experience(obs, action, reward, next_obs, done))
            agent.learn(obs, action, reward, next_obs)
            
            obs = next_obs
            total_reward += reward
            
            if done is True:
                if first_episode:
                    max_reward = total_reward
                    first_episode = False
                episode_rewards.append(total_reward)
                if total_reward > max_reward:
                    max_reward = total_reward
                print("\nEpisodio#{} finalizado con {} iteraciones. Recompensa = {}, Recompensa media = {}, Mejor recompensa = {}".
                      format(episode, step+1, total_reward, np.mean(episode_rewards), max_reward))
                if agent.memory.get_size()>100:
                    agent.replay_experience(32)
                break
    environment.close()
            
            
            
            
            
            
            
