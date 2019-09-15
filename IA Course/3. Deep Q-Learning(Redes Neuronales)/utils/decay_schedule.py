# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 21:08:01 2019

@author: sebastian villalba
"""
# Esta clase implementa un decaimiento lineal, con un valor inicial, un valor final, y el numero de pasos
class LinearDecaySchedule(object):
    #object es un objeto genérico, esta clase no hereda de ningun lado
    def __init__(self,initial_value,final_value,max_steps):
        assert initial_value > final_value, "El valor inicial debe ser estrictamente mayor que el valor final" # invariante en java
        self.initial_value = initial_value
        self.final_value = final_value
        self.decay_factor = (initial_value - final_value)/max_steps # factor de decrecimiento
        
    def __call__(self, step_num): #Cuando un método tiene __nombre__, no es un método nuestro si no un método de la clase Object que vamos a sobreescribir
        current_value = self.initial_value-step_num*self.decay_factor
        if current_value < self.final_value:
            current_value = self.final_value
        return current_value
    #Este método se llama cuando se invoca el epsilon decay con un numero de iteracion
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    epsilon_initial = 1.0
    epsilon_final = 0.005
    MAX_NUM_EPISODES = 1000
    STEPS_PER_EPISODE = 300
    linear_schedule = LinearDecaySchedule(initial_value = epsilon_initial, 
                                                final_value = epsilon_final,
                                                max_steps = 0.5*MAX_NUM_EPISODES*STEPS_PER_EPISODE) # 0.5 significa 50% del porcentaje de aprendizaje
    
    epsilons = [linear_schedule(step) for step in range(MAX_NUM_EPISODES * STEPS_PER_EPISODE)]
    plt.plot(epsilons)
    plt.show()
    