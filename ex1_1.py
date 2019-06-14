import numpy as np
import matplotlib.pyplot as plt

def kalman1d(Tb,sigmab,Tobs,sigmaobs):
    
    w = sigmab*sigmab/(sigmab*sigmab+sigmaobs*sigmaobs)
    Ta = Tb + w*(Tobs-Tb)
    sigmaa = np.sqrt((1-w)*sigmab*sigmab)
    
    return Ta, sigmaa ,w

Tb = 10.
s_b = 2.
Tt = np.random.normal(loc=Tb,scale=s_b)
s_o = 1.
To = np.random.normal(loc=Tt,scale=s_o)


Ta, s_a, w = kalman1d(Tb,s_b,To,s_o)

fig = plt.figure(figsize=(7,4))
ax = plt.axes()

T = np.arange(0, 20,0.1)
l1=ax.plot(T,1/np.sqrt(2*np.pi*s_b) * np.exp(-0.5*((T-Tb)/s_b)*((T-Tb)/s_b)),"r--",label="background dist" )
l2=ax.plot(T,1/np.sqrt(2*np.pi*s_o) * np.exp(-0.5*((T-Tt)/s_o)*((T-Tt)/s_o)), "k--", label="obs likelihood")
l3=ax.plot(T,1/np.sqrt(2*np.pi*s_a) * np.exp(-0.5*((T-Ta)/s_a)*((T-Ta)/s_a)),label="analysis dist",color = "b")
ax.axvline(Tb,color="r", linestyle="--")
ax.axvline(To,color="k", linestyle="--")
ax.axvline(Tt,label="True Temperature",color="g")
ax.axvline(Ta,color="b")

ax.legend(loc="upper left")

plt.show()

print("Tt = " + str(Tt) +"  To = " + str(To) +" Tb = " + str(Tb) +" Ta = " + str(Ta) +"      " + " sigma_a = " + str(s_a)+"      w = " + str(w))




### 1e) No the analysis is not always better, because you can just roll bad values for Tt and To. If the true value is not between the observation and the background, the analysis is always worse than the obs or background.