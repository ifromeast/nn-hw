# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 18:30:21 2017

@author: sjtu
"""

import numpy as np
from math import sin,pi,exp,cos
import random
import csv
import pandas as pd
import matplotlib.pyplot as plt

length=80
width=10
y=[0 for x in range(length)]
y[0]=1
y[1]=1
noise=[0 for x in range(length)]
for k in range(2,length):
    y[k]=0.3*y[k-1]+0.2*y[k-2]+0.5*sin(pi*cos(2*pi*k/250))+0.2*exp(-y[k-1]**2)
    
y=np.array(y)
noise=np.array(noise)

csvfile=open('rbfdata.csv','w')   #将标签数组写入文件
writer=csv.writer(csvfile)
for t in range(length):
    for s in range(width):
        xx=y[t]+random.uniform(-0.1,0.1)
        x=(t,xx)
        writer.writerow(x)
csvfile.close()

datafile = 'rbfdata.csv'
raw_data = pd.read_csv(datafile)
real_data = raw_data.values[0:,0:]
plt.figure(1)
for point in range(length*width-1):
    plt.scatter(real_data[point][0], real_data[point][1], c='y')
time=np.linspace(0,length,length)
plt.plot(time,y)
plt.xlim((5, 80))
plt.ylim((0, 1.4))   
plt.show()
