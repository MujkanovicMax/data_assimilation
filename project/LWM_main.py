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
  
    v = obs-H.dot(bg)
    x = bg + B.dot(H.transpose()).dot(np.linalg.inv(H.dot(B).dot(H.transpose())+R)).dot(v)
    
    return x, -v

def make_R_allGP(sig_h,sig_u,n):
    h_err = sig_h**2*np.ones(n)
    u_err = sig_u**2*np.ones(n)
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
    v  = np.zeros(((H.dot(bg[:,0])).shape[0],n_runs))
    
    B = make_Bmat(init, n_fc, nsteps1,nsteps2,n,dt,sig)
    R = make_R_allGP(sig_h,sig_u,n)
    R= H.dot(R.dot(H.T)) 

    obs_error = np.random.normal(0,sig_h,size=truth[:,-n_assim:].shape)
    obs = np.mean(truth[:,-n_assim:] + obs_error,axis=1)
    obs=H.dot(obs)

    bg_error = np.random.normal(0,sig_h,size=bg[:,0].shape)
    bg[:,0] = truth[:,-1] + bg_error

    an[:,0],v[:,0] = make_analyis(bg[:,0],obs,B,H,R)

    for i in range(n_runs-1):
        
        truth = np.hstack((truth,np.reshape(channel(truth[:,-1],1,sig),(2*n,1))))
        bg[:,i+1]= channel(an[:,i],1,sig)
        #init = truth[:,-1]
        #B = make_Bmat(init, n_fc, nsteps1,nsteps2,n,dt,sig)
        obs_error = np.random.normal(0,sig_h,size=truth[:,-n_assim:].shape)
        obs = np.mean(truth[:,-n_assim:] + obs_error,axis=1)
        obs=H.dot(obs)
        an[:,i+1],v[:,i+1] = make_analyis(bg[:,i+1],obs,B,H,R) 
        
    return an, bg, truth[:,-n_runs:], v

def ETKF(truth,dt,sig,sig_h,sig_u,H,n,n_runs=50, N=50,n_assim=1):
    
    an = np.zeros((2*n,n_runs,N))
    bg = np.zeros((2*n,n_runs,N))
    v  = np.zeros(((H.dot(bg[:,0])).shape[0],n_runs,N))
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
    
    indices = np.arange(0,N,1)
    B[:,:,0] = 1./(N-1) * np.sum(np.outer(bg[:,0,i]-np.mean(bg[:,0,:],axis=1), bg[:,0,i]-np.mean(bg[:,0,:],axis=1))  for i in indices) 

    Pa[:,:,0] = B[:,:,0] - B[:,:,0].dot(H.T).dot(np.linalg.inv(H.dot(B[:,:,0]).dot(H.T)+R)).dot(H).dot(B[:,:,0])  
    K[:,:,0] = Pa[:,:,0].dot(H.T).dot(np.linalg.inv(R))
    for i in range (N):
        v[:,0,i]=obs-H.dot(bg[:,0,i])
        an[:,0,i] = bg[:,0,i] + K[:,:,0].dot(v[:,0,i])
    for i in range(n_runs-1):
        truth = np.hstack((truth,np.reshape(channel(truth[:,-1],1,sig),(2*n,1))))
        for l in range (N):
            bg[:,i+1,l]= channel(an[:,i,l],1,sig)
        obs_error = np.random.normal(0,sig_h,size=truth[:,-n_assim:].shape)
        obs = np.mean(truth[:,-n_assim:] + obs_error,axis=1)
        obs=H.dot(obs)
        
        B[:,:,i+1] = 1./(N-1) * np.sum(np.outer(bg[:,i+1,j]-np.mean(bg[:,i+1,:],axis=1), bg[:,i+1,j]-np.mean(bg[:,i+1,:],axis=1))  for j in indices) 
        Pa[:,:,i+1] = B[:,:,i+1] - B[:,:,i+1].dot(H.T).dot(np.linalg.inv(H.dot(B[:,:,i+1]).dot(H.T)+R)).dot(H).dot(B[:,:,i+1])  
        K[:,:,i+1] = Pa[:,:,i+1].dot(H.T).dot(np.linalg.inv(R))
        for k in range (N):
            v[:,i+1,k]=obs-H.dot(bg[:,i+1,k])
            an[:,i+1,k] = bg[:,i+1,k] + K[:,:,i+1].dot(v[:,i+1,k])
        
    return an,bg,truth[:,-n_runs:], -v

def channel(x,dt,sig):
    '''
    Propogates state x, t timesteps forward
    '''    
    for i in range(dt):
        x = np.dot(C,x)
        x = x + alpha*np.random.normal(0,sig,1) 
    return x
############################################################parameters
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
############################################################

#Generate the model simulation
for i in range(t-1):
    truth[:,i+1] = channel(truth[:,i],dt,sig)
   
#climatology
n_ens=100
clim_ens = np.zeros((2*n, 200, n_ens))
clim = np.zeros((2*n, 200))
for j in range (n_ens):
    for i in range(199):
        clim_ens[:,i+1, j] = channel(clim_ens[:,i,j],dt,sig)
        
clim=np.mean(clim_ens, axis=2)
clim_diff = np.zeros((2*n,200,n_ens))
for i in range(n_ens):    
    clim_diff[:,:,i] = (clim_ens[:,:,i]-clim)**2
clim_diff = np.mean(clim_diff,axis = 2)


clim_cov=np.zeros((2*n, 2*n, 200, n_ens))
for i in range(200):
    for j in range(n_ens):
        clim_cov[:,:,i,j]= np.outer(clim_ens[:,i,j],clim_ens[:,i,j])
clim_covmat = np.mean(clim_cov, axis=3)  




h = truth[0:n,:]
u = truth[n:2*n,:]              #######Boundary condition is u(x = last element) = 0


###Observations
##all gridpoints obs###
sig_h =0.001
sig_u = 0.001
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

################################################## 3DVar
an_3Dvar_stat=np.zeros((2*n,50,50))
bg_3Dvar_stat=np.zeros((2*n,50,50))
truth_stat=np.zeros((2*n,50,50))
v_stat=np.zeros((2*n-1,50,50))
for i in range (50):            
    an_3Dvar, bg_3Dvar, truth_3Dvar,v_3Dvar = threeDvar (truth,dt,sig,sig_h,sig_u,H,n)
    an_3Dvar_stat[:,:,i]=an_3Dvar
    bg_3Dvar_stat[:,:,i]=bg_3Dvar
    truth_stat[:,:,i]=truth_3Dvar
    v_stat[:,:,i]=v_3Dvar
    
#an_ETKF_stat=np.zeros((2*n,50,50,50))
#bg_ETKF_stat=np.zeros((2*n,50,50,50))
#truth_stat=np.zeros((2*n,50,50))
#v_stat=np.zeros((2*n-1,50,50,50))
#an_ETKF, bg_ETKF, truth_ETKF, v_ETKF = ETKF(truth,dt,sig,sig_h,sig_u,H,n)

#for i in range (50):            
    #an_ETKF, bg_ETKF, truth_ETKF,v_ETKF = ETKF (truth,dt,sig,sig_h,sig_u,H,n)
    #an_ETKF_stat[:,:,:,i]=an_ETKF
    #bg_ETKF_stat[:,:,:,i]=bg_ETKF
    #truth_stat[:,:,i]=truth_ETKF
    #v_stat[:,:,:,i]=v_ETKF

##calculate innovation covariance matrix over time from mean over all ensembles
#v_mean = np.mean(v_stat,axis=2)
#v_cov=np.zeros((2*n-1, 2*n-1, 50, 50, 50))
#for i in range(50):
    #for k in range (50):
        #for j in range(50):
            #v_cov[:,:,i,k,j]= np.outer(v_stat[:,i,k,j],v_stat[:,i,k,j])
#v_covmat = np.mean(np.mean(v_cov, axis=4),axis=3) 

# root mean square error of analysis and background over time and space

d_an=np.sqrt(np.mean((an_3Dvar_stat-truth_stat)**2,axis=2))
print(d_an)

d_bg=np.sqrt(np.mean((bg_3Dvar_stat-truth_stat)**2,axis=2))
print(d_bg)

#d_an=np.sqrt(np.mean((np.mean(an_ETKF_stat, axis=2)-truth_stat)**2,axis=2))
#print(d_an)

#d_bg=np.sqrt(np.mean((np.mean(bg_ETKF_stat,axis=2)-truth_stat)**2,axis=2))
#print(d_bg)




diffbg = np.mean(bg_3Dvar_stat, axis=2) - truth[:,-bg_3Dvar.shape[1]:]
print(np.mean(diffbg))

diffan = np.mean(an_3Dvar_stat, axis=2) - truth[:,-an_3Dvar.shape[1]:]
print(np.mean(diffan))

#diffbg = np.mean(np.mean(bg_ETKF_stat, axis=2),axis=2) - truth[:,-bg_ETKF.shape[1]:]
#print(np.mean(diffbg))

#diffan = np.mean(np.mean(an_ETKF_stat, axis=2), axis=2) - truth[:,-an_ETKF.shape[1]:]
#print(np.mean(diffan))

######################################################

#fig,ax = plt.subplots()
#plot=ax.contourf(np.arange(0,50,1), np.arange(0,2*n,1), d_an, np.arange(np.quantile(d_an,0.4),np.quantile(d_an,0.95),1e-5))
#fig.colorbar(plot, ax=ax)
#plt.show()

fig,ax = plt.subplots()
plt.plot(np.mean(d_an,axis=0))
plt.show()

#fig,ax = plt.subplots()
#plot=ax.contourf(np.arange(0,200,1), np.arange(0,2*n,1), clim, np.arange(np.quantile(clim,0.01),np.quantile(clim,0.95),1e-5))
#fig.colorbar(plot, ax=ax)
#plt.show()



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
