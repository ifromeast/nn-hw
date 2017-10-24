import numpy as np
import matplotlib.pyplot as plt
from RBFN import RBFN
from math import sin,exp,pi,cos
import random

#x = np.linspace(0,100,10000)
#y = np.sin(x)+np.exp(-x)*2
#y = np.sin(x)
length=500
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
time=np.linspace(0,length,length)

model = RBFN(input_shape = 1, hidden_shape = 1000)
model.fit(time,y)
y_pred = model.predict(time)

plt.plot(time,y,'b-',label='real')
plt.plot(time,y_pred,'r-',label='fit')
plt.legend(loc='upper right')
plt.title('Interpolation using a RBFN')
plt.show()
