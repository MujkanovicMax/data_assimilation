import numpy as np


H2a = np.array([0,1,0])         ### H2a is a linear operator

H2b = np.array([0.5 , 0.5, 0])

H2c = np.array([0.5,0.5.0],[0,0.5,0.5])


#H(u) = H(ub) + J(ub)*(u-ub)         ###Taylor approx
    
    #J(ub)*ub = H(ub) ---> H(u) = J(ub)*u -----> 

H = 1./sqrt(ub1*ub1+ ub2*ub2 + ub3*ub3) * np.array([u1b],[u2b],[u3b])