import numpy as np


s = 5.67051*1e-8
s_appr = 5*1e-8

T1 = 10 + 10*np.sin(8*np.pi*0.5/360.)
T2 = 10 + 10*np.sin(8*np.pi*10.5/360.)
T3 = 10 + 10*np.sin(8*np.pi*100/360.)

instr_e = (s*pow(T1,4)-5.2*1e-4+s*pow(T2,4)-4.9*1e-3+s*pow(T3,4)-8*1e-4)/3      ### instrument error

lamda = np.arange(0,360,1)

T = 10 + 10*np.sin(8*np.pi*lamda/360.)

rad_grid = np.zeros(T.shape)

##################################################################################################################
##########################              Linear Interpolation    ##################################################
##################################################################################################################
for i in range(1,11):                                                                          
    
    y = (4.9*1e-3-5.2*1e-4)/10. * i + 5.2*1e-4 - (4.9*1e-3-5.2*1e-4)/10.*0.5
    rad_grid[i] = y
for i in range(11,101):
    
    y = (8*1e-4-4.9*1e-3)/(100-10.5) * i + 4.9*1e-3 - (8*1e-4-4.9*1e-3)/(100-10.5)*10.5
    rad_grid[i] = y
    
for i in range(101,361):
    
    y = (5.2*1e-4-8*1e-4)/(60.5) * i + 8*1e-4 - (5.2*1e-4-8*1e-4)/(60.5)*100
    
    if i == 360:
        i = 0
    
    rad_grid[i] = y

unr_e = np.mean(np.power(T,4)*s-rad_grid)       ### error due to unresolved scales (1° grid)
unr_e_alt = np.mean(np.power(T,4)[::3]*s-rad_grid[::3]) ### (3°grid)


op_e = np.mean(rad_grid-rad_grid/s*s_appr)      ### operator error

total_e = instr_e + unr_e + op_e                ### total error

rel_instr_e = instr_e/total_e
rel_unr_e = unr_e/total_e
rel_op_e = op_e/total_e




