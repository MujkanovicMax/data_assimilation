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

def channel(x,dt,sig):
    '''
    Propogates state x, t timesteps forward
    '''    
    for i in range(dt):
        x = np.dot(C,x)
        x = x + alpha*np.random.normal(0,sig,1) 
    return x

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
    
    #fig = plt.figure()
    #plt.imshow(B,cmap="gray")
    #plt.savefig("Bmat_3dvar.pdf")
    #plt.close("all")
    
    R = make_R_allGP(sig_h,sig_u,n)
    R= H.dot(R.dot(H.T)) 

    obs_error = np.random.normal(0,sig_h,size=truth[:,-n_assim:].shape)
    obs = np.mean(truth[:,-n_assim:] + 2.*obs_error,axis=1)
    obs=H.dot(obs)

    bg_error = np.random.normal(0,sig_h,size=bg[:,0].shape)
    bg[:,0] = truth[:,50]

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
    obs_per= np.zeros((2*n,n_runs,N))
    
    v  = np.zeros(((H.dot(bg[:,0])).shape[0],n_runs,N))
    B = np.zeros((2*n,2*n,n_runs))
    Pa = B*1
    K = np.zeros((2*n,2*n-1,n_runs))
    R = make_R_allGP(sig_h,sig_u,n)
    R= H.dot(R.dot(H.T)) 
    truth[:,-1] = channel(np.zeros(2*n),100,sig)
    obs_error = np.random.normal(0,sig_h,size=truth[:,-n_assim:].shape[0])
    
    obs = np.mean(truth[:,-n_assim:] + obs_error,axis=1)
    for i in range (N):
        obs_error = np.random.normal(0,sig_h,size=truth[:,-n_assim:].shape[0])
        obs_per[:,0,i] = obs + obs_error
        bg_error = np.random.normal(0,sig_h,size=truth[:,-1].shape[0])
        #print(bg_error)
        bg[:,0,i] = channel(np.zeros(2*n),100,sig)#truth[:,-1] + 200*bg_error
    
    
    indices = np.arange(0,N,1)
    B[:,:,0] = 1./(N-1) * np.sum(np.outer(bg[:,0,i]-np.mean(bg[:,0,:],axis=1), bg[:,0,i]-np.mean(bg[:,0,:],axis=1))  for i in indices) 
    Pa[:,:,0] = B[:,:,0] - B[:,:,0].dot(H.T).dot(np.linalg.inv(H.dot(B[:,:,0]).dot(H.T)+R)).dot(H).dot(B[:,:,0])  
    K[:,:,0] = Pa[:,:,0].dot(H.T).dot(np.linalg.inv(R))
    for i in range (N):
        v[:,0,i]=H.dot(obs_per[:,0,i])-H.dot(bg[:,0,i])
        an[:,0,i] = bg[:,0,i] + K[:,:,0].dot(v[:,0,i])
    for i in range(n_runs-1):
        truth = np.hstack((truth,np.reshape(channel(truth[:,-1],1,sig),(2*n,1))))
        
            
        obs_error = np.random.normal(0,sig_h,size=truth[:,-n_assim:].shape[0])
        obs = np.mean(truth[:,-n_assim:] + obs_error,axis=1)
        for l in range (N):
            obs_error = np.random.normal(0,sig_h,size=truth[:,-n_assim:].shape[0])
            obs_per[:,i+1,l] = obs + obs_error
            bg[:,i+1,l]= channel(an[:,i,l],1,sig)
        
        B[:,:,i+1] = 1./(N-1) * np.sum(np.outer(bg[:,i+1,j]-np.mean(bg[:,i+1,:],axis=1), bg[:,i+1,j]-np.mean(bg[:,i+1,:],axis=1))  for j in indices) 
        
        Pa[:,:,i+1] = B[:,:,i+1] - B[:,:,i+1].dot(H.T).dot(np.linalg.inv(H.dot(B[:,:,i+1]).dot(H.T)+R)).dot(H).dot(B[:,:,i+1])  
        K[:,:,i+1] = Pa[:,:,i+1].dot(H.T).dot(np.linalg.inv(R))
        K_temp =B[:,:,i+1].dot(H.transpose()).dot(np.linalg.inv(H.dot(B[:,:,i+1]).dot(H.transpose())+R))
        for k in range (N):
            v[:,i+1,k]=H.dot(obs_per[:,i+1,k])-H.dot(bg[:,i+1,k])
            
            #an[:,i+1,k] = bg[:,i+1,k] + K[:,:,i+1].dot(v[:,i+1,k])
            an[:,i+1,k] = bg[:,i+1,k] + K_temp.dot(v[:,i+1,k])
    
    
    
    return an,bg,truth[:,-n_runs:], -v,B


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
    
########################################################
   
##climatology

#n_ens=100
#clim_ens = np.zeros((2*n, 200, n_ens))
#clim = np.zeros((2*n, 200))
#for j in range (n_ens):
    #for i in range(199):
        #clim_ens[:,i+1, j] = channel(clim_ens[:,i,j],dt,sig)
        
#clim=np.mean(clim_ens, axis=2)
#clim_diff = np.zeros((2*n,200,n_ens))
#for i in range(n_ens):    
    #clim_diff[:,:,i] = (clim_ens[:,:,i]-clim)**2
#clim_diff = np.mean(clim_diff,axis = 2)


#clim_cov=np.zeros((2*n, 2*n, 200, n_ens))
#for i in range(200):
    #for j in range(n_ens):
        #clim_cov[:,:,i,j]= np.outer(clim_ens[:,i,j],clim_ens[:,i,j])
#clim_covmat = np.mean(clim_cov, axis=3)  


######################################################

##climatology plots

#fig,ax = plt.subplots()
#plot=ax.contourf(np.arange(0,50,1), np.arange(0,2*n,1), d_an, np.arange(np.quantile(d_an,0.4),np.quantile(d_an,0.95),1e-5))
#fig.colorbar(plot, ax=ax)
#plt.show()

#fig,ax = plt.subplots()
#plt.plot(np.mean(d_an,axis=0))
#plt.show()

#fig,ax = plt.subplots()
#plot=ax.contourf(np.arange(0,200,1), np.arange(0,n,1), clim[0:n,:], np.linspace(np.min(clim[0:n,:]),np.max(clim[0:n,:]),100),cmap="jet")
#ax.set_xlabel("Timestep")
#ax.set_ylabel("Gridpoint")
#ax.set_title("Climatology of height (mean)")
#fig.colorbar(plot, ax=ax)
#plt.savefig("h_climatology_mean.pdf")

#fig,ax = plt.subplots()
#plot=ax.contourf(np.arange(0,200,1), np.arange(0,n,1), clim[n:,:], np.linspace(np.min(clim[n:,:]),np.max(clim[n:,:]),100),cmap="jet")
#ax.set_xlabel("Timestep")
#ax.set_ylabel("Gridpoint")
#ax.set_title("Climatology of velocity (mean)")
#fig.colorbar(plot, ax=ax)
#plt.savefig("u_climatology_mean.pdf")

#fig,ax = plt.subplots()
#ax.imshow(clim_covmat[:,:,-1],cmap="gray")
#ax.set_title("Model Covariance Matrix")
#plt.savefig("model_covmat.pdf")

#fig,ax = plt.subplots()
#plot=ax.contourf(np.arange(0,200,1), np.arange(0,n,1), clim_diff[0:n,:], np.linspace(np.min(clim_diff[0:n,:]),0.00004,100),cmap="jet")
#ax.set_xlabel("Timestep")
#ax.set_ylabel("Gridpoint")
#ax.set_title("Climatology of height (variance)")
#fig.colorbar(plot, ax=ax)
#plt.savefig("h_climatology_variance_alt.pdf")

#fig,ax = plt.subplots()
#plot=ax.contourf(np.arange(0,200,1), np.arange(0,n,1), clim_diff[n:,:], np.linspace(np.min(clim_diff[n:,:]),0.00004,100),cmap="jet")
#ax.set_xlabel("Timestep")
#ax.set_ylabel("Gridpoint")
#ax.set_title("Climatology of velocity (variance)")
#fig.colorbar(plot, ax=ax)
#plt.savefig("v_climatology_variance_alt.pdf")

#################################################


h = truth[0:n,:]
u = truth[n:2*n,:]              #######Boundary condition is u(x = last element)an_3Dvar_stat = 0



###Observations
##all gridpoints obs###
sig_h =0.001
sig_u = 0.001
H = np.identity(2*n)
H = H[0:-1,:]

###mth gridpoint observed###
#m = 3
##h_obs = h[::m] + h_err[::m]
##u_obs = u[::m] + u_err[::m]
##R = np.diag(np.append(h_err[::m],u_err[::m]))
#H = np.zeros((int(np.ceil(n/(m+0.0))),n))
#for i in range(H.shape[0]):
    #H[i,m*i] = 1
#H = np.hstack((H,H))                                    ### mit korrelation testen, dann aber korrelation ignorieren

#### only u/h observed at every gridpont
#u_obs = u[::m] + u_err[::m]
#R = np.diag(u_err)
#H=np.identity(n)

################################################## 3DVar
#an_3Dvar_stat=np.zeros((2*n,50,50))
#bg_3Dvar_stat=np.zeros((2*n,50,50))
#truth_stat3d=np.zeros((2*n,50,50))
#v_stat3d=np.zeros((2*n-1,50,50))
#for i in range (50):  
    #x = np.zeros((2*n))
    #truth = np.empty((2*n,t))
    #truth[:,0] = x
    #for j in range(t-1):
        #truth[:,j+1] = channel(truth[:,j],dt,sig)
    #an_3Dvar, bg_3Dvar, truth_3Dvar,v_3Dvar = threeDvar (truth,dt,sig,sig_h,sig_u,H,n)
    #an_3Dvar_stat[:,:,i]=an_3Dvar
    #bg_3Dvar_stat[:,:,i]=bg_3Dvar
    #truth_stat3d[:,:,i]=truth_3Dvar
    #v_stat3d[:,:,i]=v_3Dvar
    
an_ETKF_stat=np.zeros((2*n,50,50,5))
bg_ETKF_stat=np.zeros((2*n,50,50,5))
truth_statETKF=np.zeros((2*n,50,5))
v_statETKF=np.zeros((2*n-1,50,50,5))
an_ETKF, bg_ETKF, truth_ETKF, v_ETKF,B = ETKF(truth,dt,sig,sig_h,sig_u,H,n)

for i in range (5):  
    x = np.zeros((2*n))
    truth = np.empty((2*n,t))
    truth[:,0] = x
    for j in range(t-1):
        truth[:,j+1] = channel(truth[:,j],dt,sig)
    an_ETKF, bg_ETKF, truth_ETKF,v_ETKF,B = ETKF (truth,dt,sig,sig_h,sig_u,H,n)
    an_ETKF_stat[:,:,:,i]=an_ETKF
    bg_ETKF_stat[:,:,:,i]=bg_ETKF
    truth_statETKF[:,:,i]=truth_ETKF
    v_statETKF[:,:,:,i]=v_ETKF
    
#fig,ax=plt.subplots()
#plot=plt.plot(np.arange(0,2*n),an_ETKF_stat[:,0,49,4])
#plot=plt.plot(np.arange(0,2*n),an_ETKF_stat[:,0,48,4])
#plot=plt.plot(np.arange(0,2*n),an_ETKF_stat[:,0,47,4])
#plot=plt.plot(np.arange(0,2*n),truth_statETKF[:,0,4],label="truth")
#ax.legend()

#plt.show()
########################################################
#calculate innovation covariance matrix over time from mean over all ensembles
#v_mean = np.mean(v_stat,axis=2)
#v_cov=np.zeros((2*n-1, 2*n-1, 50, 50, 50))
#for i in range(50):
    #for k in range (50):
        #for j in range(50):
            #v_cov[:,:,i,k,j]= np.outer(v_stat[:,i,k,j],v_stat[:,i,k,j])
#v_covmat = np.mean(np.mean(v_cov, axis=4),axis=3) 

#root mean square error of analysis and background over time and space

#d_an3d=np.sqrt(np.mean((an_3Dvar_stat-truth_stat3d)**2,axis=2))
#d_bg3d=np.sqrt(np.mean((bg_3Dvar_stat-truth_stat3d)**2,axis=2))
d_anETKF=np.sqrt(np.mean((np.mean(an_ETKF_stat, axis=2)-truth_statETKF)**2,axis=2))
d_bgETKF=np.sqrt(np.mean((np.mean(bg_ETKF_stat,axis=2)-truth_statETKF)**2,axis=2))





#diffbg3d = np.mean(bg_3Dvar_stat - truth_stat3d,axis=2)
#print(np.mean(diffbg3d))
#diffan3d = np.mean(an_3Dvar_stat - truth_stat3d, axis=2)
#print(np.mean(diffan3d))

#fig,ax = plt.subplots()
#plot=ax.plot(np.arange(0,50,1),  np.mean(diffan3d, axis=0),label="analysis")
#ax.set_xlabel("Time")
#ax.set_ylabel("Deviation from truth")
#ax.set_title("Analysis & Background deviation from truth of 3DVar algorithm")
#plot=ax.plot(np.arange(0,50,1),  np.mean(diffbg3d, axis=0),label="background")
#ax.legend()
#plt.tight_layout()
#plt.savefig("diff_an&bg_3DVar.pdf")



#diffbgETKF = np.mean(np.mean(bg_ETKF_stat, axis=2),axis=2) - truth[:,-bg_ETKF.shape[1]:]
#print(np.mean(diffbgETKF))

#diffanETKF = np.mean(np.mean(an_ETKF_stat, axis=2), axis=2) - truth[:,-an_ETKF.shape[1]:]
#print(np.mean(diffanETKF))


#fig,ax = plt.subplots()
#plot=ax.plot(np.arange(0,50,1),  np.mean(d_an3d[0:n,:],axis=0),label="analysis")
#ax.set_xlabel("Timestep")
#ax.set_ylabel("RMSE")
#ax.set_title("Analysis & Background RMSE of 3D Var algorithm (height)")
#plot=ax.plot(np.arange(0,50,1),  np.mean(d_bg3d[0:n,:],axis=0),label="background")
#ax.legend()
#plt.savefig("RMSE_an&bg_height_3Dvar_"+str(dt)+".pdf")

#fig,ax = plt.subplots()
#plot=ax.plot(np.arange(0,n,1), np.mean(d_an3d[n:,:],axis=1),label="analysis")
#ax.set_xlabel("Gridpoint")
#ax.set_ylabel("RMSE")
#ax.set_title("Analysis & Background RMSE of 3D Var algorithm (velocity)")
#plot=ax.plot(np.arange(0,n,1), np.mean(d_bg3d[n:,:],axis=1),label="background")
#ax.legend()
#plt.savefig("RMSE_an&bg_velocity_3Dvar_space_"+str(dt)+".pdf")

fig,ax = plt.subplots()
plot=ax.plot(np.arange(0,50,1),  np.mean(d_anETKF[0:n,:],axis=0),label="analysis")
ax.set_xlabel("Timestep")
ax.set_ylabel("RMSE")
ax.set_title("Analysis & Background RMSE of ETKF algorithm (height)")
plot=ax.plot(np.arange(0,50,1),  np.mean(d_bgETKF[0:n,:],axis=0),label="background")
ax.legend()
plt.savefig("RMSE_an&bg_height_ETKF_size100.pdf")

#fig,ax = plt.subplots()
#plot=ax.plot(np.arange(0,50,1),  np.mean(d_anETKF[n:,:],axis=0),label="analysis")
#ax.set_xlabel("Timestep")
#ax.set_ylabel("RMSE")
#ax.set_title("Analysis & Background RMSE of ETKF algorithm (velocity)")
#plot=ax.plot(np.arange(0,50,1),  np.mean(d_bgETKF[n:,:],axis=0),label="background")
#ax.legend()
#plt.savefig("RMSE_an&bg_velocity_ETKF_size100.pdf")



#############################################################################
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
