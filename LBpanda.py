#!/usr/bin/env python

#Simple diffusive lattice Boltzmann using Pandas data structures

#This was written to test my ability to implement and manipulate pandas data structures
#This simulation is extremely slow as compared to writing this in C, however, this version implements modern data science techniques
#and is designed to be used as a learning tool. It does imply basic knowledge of the lattice Boltzmann method.

#This simulation is currently implemented as a simple one dimensional diffusive system.

#I plan to modify and add functionality as I learn more,

#### Improvements ####
# 1) Remove need for global variables
# 2) Generalize to multi-dimensions
# 3) Add machine learning methods

# Kyle Strand
# ktstrandphd@gmail.com
# 10 Apr 2023

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Simulation parameters

#Global variables
iterations = 0                                          #Simulation counter
xdim = 64                                               #X dimension size
V = 3                                                   #Number of velocities, D1Q3: V = {0, 1, -1} (this can be modified for multi-dimensional application
theta = 1./3.                                           #Lattice temperature
n0 = 100                                                #Initial density
tau = 1                                                 #Relaxation time
w = pd.Series(np.arange(V))                             #array of weights constructed as a pandas series
rho = pd.Series(np.arange(xdim))                        #array of densities constructed as pandas series
f = pd.DataFrame(np.arange(V*xdim).reshape(xdim,V),     #multi-dimensional array of distribution functions constructed as a pandas DataFrame
                 index=[np.arange(xdim)],              
                 columns=np.arange(V))

#Calculates array of weights
def SetWeights():
    global w
    w[0] = 1 - theta
    w[1:3] = theta/2
    
#Initializes simulation
def Initialize():
    print("Initializing...\n")
    global f, rho
    iterations = 0                                     #Resets iterations to 0
    SetWeights()
    for i in range(xdim):
        rho[i] = n0*(1. + np.sin(2*3.14159*i/xdim))    #Initial densities in form of sine wave
        #if i < xdim/2:                                #Initial densities as step function
        #    rho[i] = 2*n0
        #else:
        #    rho[i] = 0
        f.loc[i,0] = rho[i] * w[0]                     #Setting initial distribution functions as 
        f.loc[i,1:3] = rho[i] * w[1]

#Calculates collision step
def Collision():
    global f
    for i in range(xdim):
        f[0][i] += 1./tau*(rho[i]*w[0] - f[0][i])      #Rearranging distribution functions due to collisions
        f[1][i] += 1./tau*(rho[i]*w[1] - f[1][i])
        f[2][i] += 1./tau*(rho[i]*w[2] - f[2][i])

#Move particles as defined by the velocity set
def Stream():
    global f
    f[1] = f[1].shift(1, fill_value=f[1][xdim-1])      #Periodic shifting of distribution functions by 1 position
    f[2] = f[2].shift(-1, fill_value=f[2][0])          #f[1] moves right, f[2] moves left

#Recursive algorithm
def Iteration():
    global rho
    for i in range(xdim):
        rho[i] = f[0][i] + f[1][i] + f[2][i]           #Set densities at each iteration. This can probably be done without for loop
    Collision()                                        #Perform collision step
    Stream()                                           #Perform streaming step  
    global iterations                                   
    iterations = iterations + 1                        #increment iteration counter

#Plot data using matplotlib
def PlotData():
    #Plot output
    t = np.arange(0,xdim,1)
    s = rho
    fig, ax = plt.subplots()
    ax.plot(t,s)
    plt.xlim(0,xdim)
    plt.ylim(-10,210)
    plt.show()


### Main Program ###
Initialize()
print("Running...\n")
while iterations < 5000:                              #Recursive algorithm
    Iteration()
    if (iterations % 100) == 0:
      print(f"Iteration: {iterations}\n")
PlotData()
