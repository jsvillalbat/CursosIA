# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 22:26:22 2019

@author: sebastian villalba
"""

import torch

class SLP(torch.nn.Module): #nn = Neural Network 
    """
    SLP significa Single Layer Perceptron, o neurona de una sola capa para aproximar funciones
    Esta clase recibe como párametro un modulo de red neuronal de la libreria Pytorch(Hereda de esa clase de Pytorch)
    """
    def __init__(self,input_shape,output_shape, device = torch.device("cpu")): # Método Constructor de la clase
        """
        :param input_shape: Tamaño o forma de los datos de entrada
        :param output_shape: Tamaño o forma de los datos de sálida
        :param device: El dispositivo ["cpu" o "cuda"] que la SLP debe utilizar para almacenar los inputs a cada iteracion
        """
        super(SLP,self).__init__() #Llama a la superclase y la inicializa, y lo inicializa con los párametros de la misma
        self.device = device #Hago uan copia de este device para poderlo utilizar en la propia clase
        self.input_shape = input_shape[0]
        self.hidden_shape = 40 #Unidades en la capa oculta(arbitrario)
        self.linear1 = torch.nn.Linear(self.input_shape,self.hidden_shape) #Representa la funcion de activacion RELU, este repesenta desde los datos de entrada hasta la capa oculta(RELU = Rectifier Linear Unit)
        self.out = torch.nn.Linear(self.hidden_shape,output_shape)#Repesenta desde la capa oculta hasta la salida, tambien con una funcion RELU(Rectificador), asi se devuleve la forma de la salida de forma rectificador
        
    def forward(self,x):
        """
        Rectificador dado por: f(x) = max(0,x)
        Acá se realiza la función de activación RELU, el perceptron se va a despertar si se detecta un valor positivo, de lo contrario es cero. 
        """
        x = torch.from_numpy(x).float().to(self.device)#Conversion de un array a un float y lo mapeamos al dispositivo indicado
        x = torch.nn.functional.relu(self.linear1(x))# Funcion de activación RELU(Rectificador), tomamos el valor de x que esta en CPU, le hacemos el cambio lineal y a eso la funcion RELU
        x = self.out(x) # Devolvemos la salida de la funcion lineal
        return x