# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 15:43:21 2019

@author: sebastian villalba
"""
import gym
import numpy as np

#QLearner Class
#__init(self,environment)
#discretize(self,obs)
#get_action(self,obs)
#learn(self,obs,action,reward,next_obs)
#
#EPSILON_MIN: vamos aprendiendo, mientras el incremento de aprendizaje sea superior a dicho valot
#MAX_NUM_EPISODES: número máximo de iteraciones que estamos dispuestos a realizar
#STEPS_PER_EPISODE: número máximo de pasos a realizar en cada episodio
#ALPHA: ratio de aprendizaje del agente
#GAMMA: factor de descuento del agente
#NUM_DISCRETE_BINS: número de divisiones en el caso de discretizar el espacio de estados continuo.

MAX_NUM_EPISODES = 50000
STEPS_PER_EPISODE = 200 #Este valor es propio del videojuego el Mountain Car contiene 200 episodios para pasarse el juego
EPSILON_MIN = 0.05
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN/ max_num_steps
ALPHA = 0.05
GAMMA = 0.98
NUM_DISCRETE_BINS = 30


class QLearner(object):
    
    #Metodo para inicializar las variables, es como el metodo constructor
    def __init__(self,environment):
        self.obs_shape = environment.observation_space.shape
        self.obs_high = environment.observation_space.high
        self.obs_low = environment.observation_space.low
        self.obs_bins = NUM_DISCRETE_BINS
        self.obs_width = (self.obs_high-self.obs_low)/self.obs_bins
        
        self.action_shape = environment.action_space.n
        self.Q = np.zeros((self.obs_bins+1,self.obs_bins+1,self.action_shape)) # Matriz de 31 X 31  X 3 sirve para guardar cada uno de los estados por los quev a pasando el agente
        self.apha = ALPHA
        self.gamma = GAMMA
        self.epsilon = 1.0
        
    #Método para discretizar el espacio de estados continuo
    def discretize(self,obs):
        return tuple(((obs - self.obs_low)/self.obs_width).astype(int))
        
    #Método para elegir una accion dada una observacion
    def get_action(self,obs):
        discrete_obs = self.discretize(obs)
        #Selección de la accion en base a Epsilon_Greedy
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        if np.random.random() > self.epsilon: # Con probabilidad 1-Epsilon, elegimos la mejor posible
            return np.argmax(self.Q[discrete_obs])
        else:
            return np.random.choice([a for a in range(self.action_shape)])# Con probabilidad Epsilon, elegimos al azar
  
    #Implementación de la ecuación de Bellman     
    def learn(self,obs,action,reward,next_obs):
        discrete_obs = self.discretize(obs)
        discrete_next_obs = self.discretize(next_obs)
        td_target = reward + self.gamma * np.max(self.Q[discrete_next_obs])
        td_error = td_target - self.Q[discrete_obs][action]
        self.Q[discrete_obs][action] += self.apha*td_error
        
#Método para entrenar a nuestro agente
        
def train(agent,environment):
    best_reward = -float("inf")
    for episode in range(MAX_NUM_EPISODES):
        done = False
        obs = environment.reset()
        total_reward = 0.0
        while not done:
            action = agent.get_action(obs) # agent es de la clase QLearner
            next_obs,reward,done,info = environment.step(action)
            agent.learn(obs,action,reward,next_obs)
            obs = next_obs
            total_reward +=reward
        if total_reward > best_reward:
            best_reward = total_reward
        print("El Episodio número {} con recompensa: {}, mejor recompensa: {}, epsilon: {}".format(episode,total_reward,best_reward,agent.epsilon))
    
    #De todas las politicas de entrenamiento que hemos obtenido, devolvemos la mejor de todas
    return np.argmax(agent.Q, axis = 2)
            
def test(agent, environment,policy):
    done = False
    obs = environment.reset()
    total_reward = 0.0
    while not done:
        action = policy[agent.discretize(obs)] #Accionq ue dictamina la policitca que hemos entrenado
        next_obs,reward,done,info = environment.step(action)
        obs = next_obs
        total_reward +=reward
    return total_reward

if __name__ == "__main__":
    environment = gym.make("MountainCar-v0")
    agent = QLearner(environment)
    learned_policy = train(agent,environment)
    monitor_path = "./monitor_output"
    environment = gym.wrappers.Monitor(environment,monitor_path, force = True)
    for _ in range(1000):
        test(agent, environment,learned_policy)
    environment.close()

#Funciona bien al ejecutar, solo falta una dependencia ffmpeg o libav-tools que en Windows es dificil de instalar, revisar esto
    