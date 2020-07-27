#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 14:03:02 2020

@author: cfd
"""

import wget
import numpy as np
import matplotlib.pyplot as plt
import h5py
import PP
#import Plots
import importlib
importlib.reload(PP)
#importlib.reload(Plots)

from PP import extract_grid
from PP import Read_solution
from PP import Read_solution_all
from PP import Read_stats
from PP import Read_stats_raw
from PP import MeanXZ
from PP import MeanZ
from PP import wallunit
#from Plots import plot_2d

#droplink='https://www.dropbox.com/sh/ak1mxjh6aq8isr0/AAASxtrP7cbkpgJdUmjnV7t5a/Channel_180/'

rhom = 0.014
tauw = 5.92464e-05
utau=(tauw/rhom)**0.5
mu = 5.05967e-06
Retau=(rhom/mu)*utau

#%%  Download mesh
url='https://www.dropbox.com/sh/ak1mxjh6aq8isr0/AADZo4zHreHbX56SeqZIfGRca/Channel_180/setup/Channel-180.pyfrm?dl=1'
wget.download(url,out='Channel-180.pyfrm') 

#%%      Mesh

Mesh,Lx,Ly,Lz =extract_grid()
X,Y,Z=np.meshgrid(Lx,Ly,Lz, indexing='ij')
yp=(Ly[:int(1+np.floor(len(Ly)/2))]+1)*Retau

#%%
timesteps=np.arange(2050,6400,50)
n=29

#U=Read_solution(timesteps[n])

rho,rhou,rhov,rhow,E=Read_solution_all(timesteps[n])
#%%

U=rhou/rho
#plt.pcolormesh(np.squeeze(X[:,0,:]), np.squeeze(Z[:,0,:]), np.squeeze(U[:,8,:]), cmap='jet')
cs=plt.imshow(np.squeeze(U[:,0,:]).T,extent=[np.squeeze(X[0,0,0]), np.squeeze(X[-1,0,0]) , np.squeeze(Z[0,0,0]), np.squeeze(Z[0,0,-1])], cmap='jet', origin='lower', interpolation='spline16')
plt.colorbar(cs)

#%%
cs=plt.imshow(np.squeeze(U[:,:,150]).T,extent=[np.squeeze(X[0,0,150]), np.squeeze(X[-1,0,150]) , np.squeeze(Y[0,0,150]), np.squeeze(Y[0,-1,150])], cmap='jet', origin='lower', interpolation='spline16')
plt.colorbar(cs)
#%%
cs=plt.pcolormesh(np.squeeze(Z[150,:,:]), np.squeeze(Y[150,:,:]), np.squeeze(U[150,:,:]), cmap='jet')
plt.colorbar(cs)

#%%




