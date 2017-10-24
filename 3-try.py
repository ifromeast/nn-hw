# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 02:17:19 2017

@author: sjtu
"""
#测试数据

import numpy as np
import matplotlib.pyplot as plt
from math import sin,pi,exp,cos
import random

length=1000
y=[0 for x in range(length)]
y[0]=1
y[1]=1
noise=[0 for x in range(length)]
for k in range(2,length):
    y[k]=0.3*y[k-1]+0.6*y[k-2]+0.6*sin(pi*cos(2*pi*k/250))+0.9*exp(-y[k-1]**2)
    noise[k]=random.uniform(-0.1,0.1)
    
y=np.array(y)
noise=np.array(noise)
yy=y+noise  


plt.figure(1)
time=np.linspace(0,length,length)
plt.plot(time,y)
plt.figure(2)
plt.plot(time,noise)
plt.figure(3)
plt.plot(time,yy)
plt.show()