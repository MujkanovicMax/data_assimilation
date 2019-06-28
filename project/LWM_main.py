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
  
    
    x = bg + B.dot(H.transpose()).dot(np.linalg.inv(H.dot(B).dot(H.transpose())+R)).dot(obs-H.dot(bg))
    
    return x

def make_R_allGP(sig_h,sig_u,n):
    h_err = sig_h*np.ones(n)
    u_err = sig_u*np.ones(n)
    R=np.diag(np.append(h_err, u_err))
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

def threeDvar(truth,dt,sig,sig_h,sig_u,H,n,n_runs=50,n_assim=1,n_fc=200,nsteps1=10,nsteps2=40):

    init = truth[:,-1]

    an = np.zeros((2*n,n_runs))
    bg = np.zeros((2*n,n_runs))

    B = make_Bmat(init, n_fc, nsteps1,nsteps2,n,dt,sig)
    R = make_R_allGP(sig_h,sig_u,n)
    R= H.dot(R.dot(H.T)) 

    obs_error = np.random.normal(0,sig_h,size=truth[:,-n_assim:].shape)
    obs = np.mean(truth[:,-n_assim:] + obs_error,axis=1)
    obs=H.dot(obs)

    bg_error = np.random.normal(0,sig_h,size=bg[:,0].shape)
    bg[:,0] = truth[:,-1] + bg_error

    an[:,0] = make_analyis(bg[:,0],obs,B,H,R)

    for i in range(n_runs-1):
        
        truth = np.hstack((truth,np.reshape(channel(truth[:,-1],1,sig),(2*n,1))))
        bg[:,i+1]= channel(an[:,i],1,sig)
        #init = truth[:,-1]
        #B = make_Bmat(init, n_fc, nsteps1,nsteps2,n,dt,sig)
        obs_error = np.random.normal(0,sig_h,size=truth[:,-n_assim:].shape)
        obs = np.mean(truth[:,-n_assim:] + obs_error,axis=1)
        obs=H.dot(obs)
        an[:,i+1] = make_analyis(bg[:,i+1],obs,B,H,R) 
        
    return an, bg, truth[:,-n_runs:]

def ETKF(truth,dt,sig,sig_h,sig_u,H,n,n_runs=50, N=50,n_assim=1):
    
    an = np.zeros((2*n,n_runs,N))
    bg = np.zeros((2*n,n_runs,N))
    B = np.zeros((2*n,2*n,n_runs))
    Pa = B*1
    K = np.zeros((2*n,2*n-1,n_runs))
    R = make_R_allGP(sig_h,sig_u,n)
    R= H.dot(R.dot(H.T)) 
    
    obs_error = np.random.normal(0,sig_h,size=truth[:,-n_assim:].shape)
    obs = np.mean(truth[:,-n_assim:] + obs_error,axis=1)
    obs=H.dot(obs)
    for i in range (N):
        bg_error = np.random.normal(0,sig_h,size=truth[:,-1].shape)
        bg[:,0,i] = truth[:,-1] + bg_error
    print(bg_error)
    
    indices = np.arange(0,N,1)
    print("1")
    B[:,:,0] = 1./(N-1) * np.sum(np.outer(bg[:,0,i]-np.mean(bg[:,0,:],axis=1), bg[:,0,i]-np.mean(bg[:,0,:],axis=1))  for i in indices) 
    print("2")

    Pa[:,:,0] = B[:,:,0] - B[:,:,0].dot(H.T).dot(np.linalg.inv(H.dot(B[:,:,0]).dot(H.T)+R)).dot(H).dot(B[:,:,0])  
    print("3")
    K[:,:,0] = B[:,:,0].dot(H.T).dot(np.linalg.inv(R))
    print("4")
    for i in range (N):
        an[:,0,i] = bg[:,0,i] + K[:,:,0].dot(obs-H.dot(bg[:,0,i]))
    for i in range(5):#n_runs-1
        truth = np.hstack((truth,np.reshape(channel(truth[:,-1],1,sig),(2*n,1))))
        print("a")
        for l in range (N):
            bg[:,i+1,l]= channel(an[:,i,l],1,sig)
        print("b")
        obs_error = np.random.normal(0,sig_h,size=truth[:,-n_assim:].shape)
        print("c")
        obs = np.mean(truth[:,-n_assim:] + obs_error,axis=1)
        print("d")
        obs=H.dot(obs)
        print("e")
        
        B[:,:,i+1] = 1./(N-1) * np.sum(np.outer(bg[:,i+1,j]-np.mean(bg[:,i+1,:],axis=1), bg[:,i+1,j]-np.mean(bg[:,i+1,:],axis=1))  for j in indices) 
        #print(B)
        Pa[:,:,i+1] = B[:,:,i+1] - B[:,:,i+1].dot(H.T).dot(np.linalg.inv(H.dot(B[:,:,i+1]).dot(H.T)+R)).dot(H).dot(B[:,:,i+1])  
        print("g")
        K[:,:,i+1] = B[:,:,i+1].dot(H.T).dot(np.linalg.inv(R))
        for k in range (N):
            an[:,i+1,k] = bg[:,i+1,k] + K[:,:,i+1].dot(obs-H.dot(bg[:,i+1,k]))
        print("h")
        print("an ")
        print(an[:,i+1,25])
        print("bg")
        print(bg[:,i+1,25])
    print("6")    
    return an,bg,truth

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


sig_h = np.mean(np.abs(u))*0.01
sig_u = np.mean(np.abs(h))*0.01
H = np.identity(2*n)
H = H[0:-1,:]



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

an_3Dvar_stat=np.zeros((2*n,50,50))
bg_3Dvar_stat=np.zeros((2*n,50,50))
truth_stat=np.zeros((2*n,50,50))
for i in range (50):            
    an_3Dvar, bg_3Dvar, truth_3Dvar = threeDvar (truth,dt,sig,sig_h,sig_u,H,n)
    an_3Dvar_stat[:,:,i]=an_3Dvar
    bg_3Dvar_stat[:,:,i]=bg_3Dvar
    truth_stat[:,:,i]=truth_3Dvar
    

#an_ETKF, bg_ETKF, truth_ETKF = ETKF(truth,dt,sig,sig_h,sig_u,H,n)


d_an=np.sqrt(np.mean((an_3Dvar_stat-truth_stat)**2,axis=2))
print(d_an)

d_bg=np.sqrt(np.mean((bg_3Dvar_stat-truth_stat)**2,axis=2))
print(d_bg)


#diffbg = bg_3Dvar - truth[:,-bg_3Dvar.shape[1]:]
#print(np.mean(diffbg))

#diffan = an_3Dvar - truth[:,-an_3Dvar.shape[1]:]
#print(np.mean(diffan))



fig,ax = plt.subplots()
plot=ax.contourf(np.arange(0,50,1), np.arange(0,2*n,1), d_an, np.arange(np.min(d_an),np.quantile(d_an,0.95),1e-5))
fig.colorbar(plot, ax=ax)
plt.show()

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
