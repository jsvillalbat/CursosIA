# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 21:15:32 2019

@author: sebastian villalba
"""

temp = 32
while temp < 73:
    celsius = (temp - 32)*(5/9)
    print("F°: ",temp,"       C°: ",celsius)
    temp+=1
  
a = 2
for i in range(1,4):
    a = i ** a
print(a)
    