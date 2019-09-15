# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
import gym as gm #Importamos la libreria de OpenAIGym
enviroment = gm.make("CartPole-v0") # Lanzamos o creamos un ambiente de la montaña Rusa
enviroment.reset() #Limpiamos y preparamos el entorno para tomar decisiones
for _ in range(2000): # Durante 2000 iteraciones
    enviroment.render() #Pintamos en pantalla la acción
    enviroment.step(enviroment.action_space.sample()) #Tomamos una decision aleatoria
    # next_step, o proximo estado, es un objeto de Python
    #reward o recompensa, rd un numero flotante, recompensa de la acción anterior
    #done, es true o false si ya ha finalizado el episodio,booleano
    #info, información adicional que no usa el agente, pero es util para nosotros, es un diccionario
    
enviroment.close() #Cierra el ambiente creada o la sesion de OpenAiGym, después de las 2000 iteraciones


