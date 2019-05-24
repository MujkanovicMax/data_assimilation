# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 18:43:23 2017

@author: Yvonne.Ruckstuhl
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as anim
import math

def MatrixC(n):
    '''
    Generates the matrix corresponding to the linear model. 
    Please treat this as a black box.
    '''
    
    c = 0.0002
    D = 10.
    g = 9.81
    L = 60000.0
    dx = L/(n+0.5)
    dt = 10*60.0
    phi = 0.85
    
    A = np.zeros((2*n,2*n))
    A[1:n,1:n] = 1./dt*np.identity(n-1) 
    A[n:2*n-1,n:2*n-1] = (1./dt+0.5*c)*np.identity(n-1) 
    A[n:2*n-1,1:n] = g/(2.0*dx)*np.identity(n-1) 
    A[1:n,n:2*n-1] = -D/(2.0*dx)*np.identity(n-1) 
    for i in range(1,n-1):
        A[i,n+i] = D/(2.*dx)  
        A[n+i,i] = -g/(2.*dx) 
    A[1,0]=-0.5*g/dx
    A[0,0]=1.0
    C = np.copy(A)
    C[n:2*n-1,n:2*n-1] = (1./dt-0.5*c)*np.identity(n-1)
    C[n:2*n-1,1:n] = -A[n:2*n-1,1:n]
    C[1:n,n:2*n-1] = -A[1:n,n:2*n-1]
    C[1,0]=0.5*g/dx
    C[0,0]=phi
    
    A[0:2*n-1,0:2*n-1]=np.linalg.inv(A[0:2*n-1,0:2*n-1])
    C = np.dot(A,C)
    
    alpha = np.zeros((2*n))
    alpha[0]=1.0
    alpha = np.dot(A,alpha)
    return C, alpha
     


    

