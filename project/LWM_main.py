#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 14:22:30 2019

@author: Yvonne.Ruckstuhl
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as anim
from matrix import MatrixC

def make_analyis(bg,obs,B,H,R):
    
    x = bg + B.dot(H.transpose()).dot(np.linalg.inv(H.dot(B).dot(H.transpose())+R)).dot(H.dot(bg)-obs)
    return x

def make_R_allGP(sv,sig_h,sig_u,n):
    h = sv[0:n]
    u = sv[n:]
    h_err = h*np.random.normal(0,sig_h)
    h_obs = h + h_err

    u_err = u*np.random.normal(0,sig_u)
    u_obs = u + u_err

    R = np.diag(np.append(h_err,u_err))
    return R

def make_Bmat(init, n_fc, nsteps1,nsteps2,n,dt,sig):
    
    tmp_B1 = np.ones((2*n,n_fc))
    tmp_B2 = np.ones((2*n,n_fc))
    for i in range(n_fc):
        for j in range(nsteps1):
            init = channel(init,dt,sig)             
        tmp_B1[:,i]=init   
    for i in range(n_fc):
        init=tmp_B1[:,i]
        for j in range(nsteps2):
            init = channel(init,dt,sig)
        tmp_B2[:,i]=init 

    diff = tmp_B2-tmp_B1
    B=np.zeros((2*n,2*n))
    for i in range (n_fc):
        B=B+np.outer(diff[:,i],diff[:,i])
    B = B/n_fc
    return B



def channel(x,dt,sig):
    '''
    Propogates state x, t timesteps forward
    '''    
    for i in range(dt):
        x = np.dot(C,x)
        x = x + alpha*np.random.normal(0,sig,1) 
    return x

n = 40 # number of grid point
C,alpha = MatrixC(n) 

#Define the standard deviation of the model error
sig = 0.01

#Define an intial model state, where n is the number of grid points defined above
x = np.zeros((2*n))

#Define the number of model time steps to compute without saving the output
dt = 1

#Define the length of the model simulation
t = 100 #(total length is dt*t)

#Allocate an array to store the model simulation
truth = np.empty((2*n,t))
truth[:,0] = x

###divergence test
#truth_alt = truth*1

#Generate the model simulation
for i in range(t-1):
    truth[:,i+1] = channel(truth[:,i],dt,sig)
    #truth_alt[:,i+1] = channel(truth_alt[:,i],dt,sig)
### Anmerkung: truth enthält von 0 bis n-1 die höhe und von n bis 2n-1 die geschwindigkeit u

h = truth[0:n,:]
u = truth[n:2*n,:]              #######Boundary condition is u(x = last element) = 0

#h = h[::-1,:]
#u = u[::-1,:]

#truth = np.vstack((h,u))
#h_alt = truth_alt[0:n,:]
#u_alt = truth_alt[n:2*n,:] 

###Observations
##all gridpoints obs###


sig_h = 0.01
sig_u = 0.01
R = make_R_allGP(truth[:,-1],sig_h,sig_u,n)
H = np.identity(2*n)


###mth gridpoint observed###
#m = 3
#h_obs = h[::m] + h_err[::m]
#u_obs = u[::m] + u_err[::m]
#R = np.diag(np.append(h_err[::m],u_err[::m]))
#H = np.zeros((int(np.ceil(n/(m+0.0))),n))
#for i in range(H.shape[0]):
    #H[i,m*i] = 1
#H = np.hstack((H,H))                                    ### mit korrelation testen, dann aber korrelation ignorieren

#### only u/h observed at every gridpont
#u_obs = u[::m] + u_err[::m]
#R = np.diag(u_err)
#H=np.identity(n)

###B matrix

n_runs = 50

truth_flip = np.vstack((truth[0:n,:][::-1,:],truth[n:,:][::-1,:]))
init = truth_flip[:,-1]
n_fc = 200
nsteps1 = 10
nsteps2 = 40
n_assim = 5

an = np.zeros((2*n,n_runs))

B = make_Bmat(init, n_fc, nsteps1,nsteps2,n,dt,sig)


obs_error = truth_flip[:,-n_assim:]* np.random.normal(0,sig_h,size=truth_flip[:,-n_assim:].shape)
obs = np.mean(truth_flip[:,-n_assim:] + obs_error,axis=1)

bg_error = truth_flip[:,-n_assim:]* np.random.normal(0,sig_h,size=truth_flip[:,-n_assim:].shape)
bg = np.mean(truth_flip[:,-n_assim:] + bg_error,axis=1)

an[:,0] = make_analyis(bg,obs,B,H,R)

for i in range(n_runs-1):
    
    truth = np.hstack((truth,np.reshape(channel(truth[:,-1],1,sig),(2*n,1))))
    
    truth_flip = np.vstack((truth[0:n,:][::-1,:],truth[n:,:][::-1,:]))
    init = truth_flip[:,-1]
    
    B = make_Bmat(init, n_fc, nsteps1,nsteps2,n,dt,sig)
    R = make_R_allGP(init,sig_h,sig_u,n)
    obs_error = truth_flip[:,-n_assim:]* np.random.normal(0,sig_h,size=truth_flip[:,-n_assim:].shape)
    obs = np.mean(truth_flip[:,-n_assim:] + obs_error,axis=1)
    bg = an[:,i]
    an[:,i+1] = make_analyis(bg,obs,B,H,R)
    
    

#fig = plt.figure()
#plt.imshow(B*1000,cmap="gray")
#plt.show()

#Generate a visual simulation of the two variables
#f, (ax1, ax2) = plt.subplots(2, sharex=True,figsize=(15,15))
#ims=[]
#for time in range(0,t):
    
    ##ax1.set_title('Height h',fontdict=None,fontsize=24)
    ##im, = ax1.plot(truth[0:n,time],'g',lw=2.5, label='truth')
    ##ax1.tick_params(labelsize=18)
   
    ##ax2.set_title('Velocity u',fontdict=None,fontsize=24)
    ##im2, = ax2.plot(truth[n:2*n,time],'g',lw=2.5)
    ##ax2.tick_params(labelsize=18)
    
    ##f.subplots_adjust(hspace=0.15)
    ##plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

    ##ims.append([im,  im2, ])
    
    #ax1.set_title("Height diff between forecasts" , fontsize=24)
    #im, = ax1.plot(h[:,time]-h_alt[:,time],"g",lw=2.5)
    #ax1.tick_params(labelsize=18)
    
    #ax2.set_title("Velocity diff between forecasts" , fontsize=24)
    #im2, = ax2.plot(u[:,time]-u_alt[:,time],"g",lw=2.5)
    #ax1.tick_params(labelsize=18)
    
    #f.subplots_adjust(hspace=0.15)
    #plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

    #ims.append([im,  im2, ])
    
#ani = anim.ArtistAnimation(f,ims)
#ani.save('LWM_differences.mp4')
