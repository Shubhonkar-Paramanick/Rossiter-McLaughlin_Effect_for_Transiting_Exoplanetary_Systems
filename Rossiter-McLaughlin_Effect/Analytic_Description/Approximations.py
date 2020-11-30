#!/usr/bin/env python3
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12,7]
zhe=0.1
gani = np.linspace(0,1-zhe,1000)
I1 = ((1/(16*(1-(gani**2))**(3/2)))*(-(8*gani**6)+(31*gani**4)-(36*gani**2)+16)) + (((zhe**2)/(256*(1-(gani**2))**(7/2)))*(-(112*gani**8)+(888*gani**6)+(2984*gani**4)-(2880*gani**2)+896)) + (((zhe**4)/(6144*(1-(gani**2))**(11/2)))*((9288*gani**8)-(45096*gani**6)+(78432*gani**4)-(56448*gani**2)+13824))
plt.plot(gani,I1, 'k', label='')
plt.show()

---------------------------------------------------------------------------------------------------------------------------------------------------------------------


gamma = 0.3
rho = np.linspace(0,1-gamma,1000)
W1_Ohta = ((1-(rho**2))**0.5) - ((gamma**2)*((2-(rho**2))/(8*(1-(rho**2))**1.5)))
plt.plot(rho,W1_Ohta, 'k--', label='')
plt.show()


zhe = 0.3
gani = np.linspace(0,1-zhe,1000)
W1_My_Complete = ((2-(4*gani**2)+(3*gani**4)-(gani**6))/(2*((1-(gani**2))**1.5)))+(((zhe**2)*(-(2*gani**8)+(7*gani**6)-(10*gani**4)+(7*gani**2)-2))/(8*((1-(gani**2))**3.5)))+(((zhe**4)*((gani**8)-(10*gani**6)+(9*gani**4)+(8*gani**2)+8))/(192*((1-(gani**2))**5.5)))
plt.plot(rho,W1_My_Complete, 'g--', label='')
plt.show()




plt.plot(rho,W1_My, 'g--', label='')
plt.plot(rho,W1_Ohta, 'k--', label='')
plt.show()





---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

gamma = 0.3
rho = np.linspace(0,1-gamma,1000)
WA_Ohta = ((2*((1-(rho**2))**0.5)*(1-(0.25*rho**2)))/(3*gamma**2))-(((15*rho**4)-(28*rho**2)+(16))/(16*((1-(rho**2))**1.5)))-(gamma**2*(((17*rho**6)-(64*rho**4)+(104*rho**2)-32)/(128*((1-(rho**2))**3.5))))
plt.plot(rho,WA_Ohta, 'k--', label='')
plt.show()




zhe = 0.3
gani = np.linspace(0,1-zhe,1000)

WA_My = ((2*((1-(gani**2))**1.5))/(3*zhe**2))-((2-(3*gani**2))/(2*((1-(gani**2))**0.5)))+(((zhe**2)*((3*gani**4)-(8*gani**2)+8))/(32*((1-(gani**2))**2.5)))+(((zhe**4)*((gani**6)-(6*gani**4)+(24*gani**2)+16))/(384*((1-(gani**2))**4.5)))+(((zhe**6)*((3*gani**8)-(32*gani**6)+(288*gani**4)+(768*gani**2)+128))/(8192*((1-(gani**2))**6.5))) + (((gani**2)*((1-(gani**2))**0.5))/(2*zhe**2))-(((12*gani**2)-(9*gani**4))/(16*((1-(gani**2))**1.5)))-(((zhe**2)*((10*gani**6)-(40*gani**4)+(80*gani**2)))/(256*((1-(gani**2))**3.5)))-(((zhe**4)*((7*gani**8)-(56*gani**6)+(336*gani**4)+(448*gani**2)))/(2048*((1-(gani**2))**5.5)))-(((zhe**6)*((27*gani**10)-(360*gani**8)+(4320*gani**6)+(17280*gani**4)+(5760*gani**2)))/(32768*((1-(gani**2))**7.5)))

plt.plot(rho,WA_My, 'g--', label='')
plt.show()




plt.plot(rho,WA_My, 'g--', label='')
plt.plot(rho,WA_Ohta, 'k--', label='')
plt.show()

WA_Error_Ohta = (WA_My/WA_Ohta)-1
plt.plot(rho,WA_Error_Ohta, 'y--', label='')
plt.show()

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





gamma = 0.3
rho = np.linspace(0,1-gamma,1000)
WB_Ohta = -(((rho**2)*(1-(0.25*rho**2)))/(4*(1-(rho**2))**1.5))-(((gamma**2)*(rho**2)*(24+(rho**4)))/(128*(1-(rho**2))**3.5))
plt.plot(rho,WB_Ohta, 'k--', label='')
plt.show()




zhe = 0.3
gani = np.linspace(0,1-zhe,1000)
WB_My = -(((gani**2)*(1-(0.25*gani**2)))/(4*(1-(gani**2))**1.5))-(((zhe**2)*((24*gani**2)+(gani**6)))/(128*(1-(gani**2))**3.5))-(((zhe**4)*((5*gani**8)+(40*gani**6)+(1200*gani**4)+(960*gani**2)))/(6144*(1-(gani**2))**5.5))
plt.plot(rho,WB_My, 'b--', label='')
plt.show()




plt.plot(rho,WB_My, 'b--', label='')
plt.plot(rho,WB_Ohta, 'k--', label='')
plt.show()

WB_Error_Ohta = (WB_My/WB_Ohta)-1
plt.plot(rho,WB_Error_Ohta, 'y--', label='')
plt.show()
----------------------------------------------------------------------------------------------------------------------------------------------------------------



gamma = 0.3
rho = np.linspace(0,1-gamma,1000)
W1_Ohta = (((2*((1-(rho**2))**0.5))*(1-(0.25*rho**2)))/(3*gamma**2))-WA_Ohta+WB_Ohta
plt.plot(rho,W1_Ohta, 'k--', label='')
plt.show()


zhe = 0.3
gani = np.linspace(0,1-zhe,1000)
W1_My = (((2*((1-(gani**2))**0.5))*(1-(0.25*gani**2)))/(3*zhe**2))-WA_My+WB_My
plt.plot(rho,W1_My, 'g--', label='')
plt.show()




plt.plot(rho,W1_My, 'g--', label='')
#plt.plot(rho,W1_My_Complete, 'c--', label='')
plt.plot(rho,W1_Ohta, 'k--', label='')
plt.show()

W1_Error_Ohta = (W1_My/W1_Ohta)-1
plt.plot(rho,W1_Error_Ohta, 'y--', label='')
plt.show()





-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#WC_My = (((96*gani**4)-(96*gani**2))/(128*((1-(gani**2))**1.5))) + (((zhe**2)*((-16*gani**9) + (15228*gani**8) + (80*gani**7) - (39120*gani**6) - (192*gani**5) + (57792*gani**4) + (128*gani**3) - (38016*gani**2) + 9216))/(4096*(gani**2)*((1-(gani**2))**3.5)))+(((zhe**4)*((24*gani**11) + (740328*gani**10) - (88*gani**9) - (3451400*gani**8) - (320*gani**7) + (6490784*gani**6) - (1152*gani**5) - (6129792*gani**4) + (1536*gani**3) + (2900992*gani**2) - 550912))/(49152*(gani**2)*((1-(gani**2))**5.5)))

WC_My = ((176*gani**4 - 304*gani**2 + 128)/(128*((1-(gani**2))**1.5))) + (((zhe**2)*(-16*gani**7 + 1232*gani**6 + 80*gani**5 - 4240*gani**4 - 192*gani**3 + 5184*gani**2 + 128*gani - 2176))/(4096*((1-(gani**2))**3.5)))+(((zhe**4)*(24*gani**9 - 456*gani**8 - 88*gani**7 + 2568*gani**6 - 320*gani**5 - 5568*gani**4 - 1152*gani**3 + 11136*gani**2 + 1536*gani - 7680))/(49152*((1-(gani**2))**5.5)))

W2_My = WC_My + W1_My


plt.plot(rho,W1_My, 'g--', label='')
plt.plot(rho,W2_My, 'k--', label='')
plt.show()


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


from scipy.integrate import quad

integrand_one= lambda phi, gamma, rho: ((1-(rho**2)-(gamma**2)-(2*rho*gamma*np.cos(phi)))**1.5)
I1 = lambda rho, gamma: ((1)/(3*(np.pi)*(gamma**2)))*quad(integrand_one, 0, 2*np.pi, args=(rho, gamma))[0]
gamma = 0.3
rho = np.linspace(0, 1-gamma, 1000)
integrand_one_out = np.vectorize(I1)(rho, gamma)  # some values of gamma
plt.plot(rho, integrand_one_out,'k--')
plt.show()


integrand_two= lambda phi, gamma, rho: ((((rho**2)*(np.cos(phi)))+(rho*gamma))*(np.cos(phi))*((1-(rho**2)-(gamma**2)-(2*rho*gamma*np.cos(phi)))**0.5))
I2 = lambda rho, gamma: ((1)/(2*(np.pi)*(gamma**2)))*quad(integrand_two, 0, 2*np.pi, args=(rho, gamma))[0]
rho = np.linspace(0, 1-gamma, 1000)
integrand_two_out = np.vectorize(I2)(rho, gamma)  # some values of gamma
plt.plot(rho, integrand_two_out,'k--')
plt.show()


integrand_wa = integrand_one_out+integrand_two_out
plt.plot(rho, integrand_wa,'r--')
plt.plot(rho,WA_Ohta, 'k--', label='')
plt.plot(rho,WA_My, 'g', label='')
plt.show()


integrand_three= lambda phi, gamma, rho: ((rho*np.cos(phi))*(1-((rho*np.sin(phi))**2)))*((np.arcsin((rho*np.cos(phi))/(np.sqrt(1-((rho*np.sin(phi))**2)))))-(np.arcsin(((gamma)+(rho*np.cos(phi)))/(np.sqrt(1-((rho*np.sin(phi))**2))))))
I3 = lambda rho, gamma: ((1)/(2*(np.pi)*(gamma**2)))*quad(integrand_three, 0, 2*np.pi, args=(rho, gamma))[0]
rho = np.linspace(0, 1-gamma, 1000)
integrand_wb = np.vectorize(I3)(rho, gamma)  # some values of gamma
plt.plot(rho, integrand_wb,'r--')
plt.plot(rho,WB_Ohta, 'k--', label='')
plt.plot(rho,WB_My, 'g--', label='')
plt.show()


W1_Ohta_Numerical = (((2*((1-(rho**2))**0.5))*(1-(0.25*rho**2)))/(3*gamma**2))-integrand_wa+integrand_wb
plt.plot(rho,W1_Ohta_Numerical, 'r--', label='')
plt.plot(rho,W1_My, 'g--', label='')
plt.plot(rho,W1_Ohta, 'k--', label='')
plt.show()





---------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#For WA second integrand choosing the function

gamma = 0.3
rho = np.linspace(0, 1-gamma, 1000)
phi = 3*np.pi/4
'''Wrong, When cos(phi) is negative'''
two = (((1-(2*rho*gamma*np.cos(phi)))*((((rho*np.cos(phi))**2)+(rho*gamma*np.cos(phi)))**2))-((((rho**3)*(np.cos(phi))*(np.cos(phi)))+((rho**2)*gamma*np.cos(phi)))**2)-((((rho**2)*gamma*(np.cos(phi))*(np.cos(phi)))+(rho*(gamma**2)*np.cos(phi)))**2))**0.5

'''Correct'''                              
one = ((((rho**2)*(np.cos(phi)))+(rho*gamma))*(np.cos(phi))*((1-(rho**2)-(gamma**2)-(2*rho*gamma*np.cos(phi)))**0.5))
plt.plot(rho, one,'g.')                                                
plt.plot(rho, two,'r--')                                               
plt.show()





b = np.array([0,np.pi/6,np.pi/4,np.pi/3,np.pi/2,2*np.pi/3,3*np.pi/4,5*np.pi/6,np.pi,7*np.pi/6,5*np.pi/4,4*np.pi/3,3*np.pi/2,5*np.pi/3,7*np.pi/4,11*np.pi/6,2*np.pi])
a = np.cos(b)
c = np.zeros((1000,17))
phi = 3*np.pi/4
gamma = 0.3
rho = np.array([np.linspace(0, 1-gamma, 1000)]).T
for i in range(0,17):
    for j in range(0,1000):
        c[j][i] = ((((rho[j])**2)*((a[i])**2))+((rho[j])*gamma*(a[i])))*((1-((rho[j])**2)-(gamma**2)-(2*(rho[j])*gamma*(a[i])))**0.5)
    
plt.plot(rho,c[:,7],'r--')
plt.show()




#SYMPY

import sympy  
from sympy import expand, symbols
from sympy import pretty_print as pp, latex
s,d,c = sympy.symbols('s d c')
delete = ((s**2)+(2*s*d*c))
pp(delete.expand(), use_unicode=True)





-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# WC2 Odd or Even
a =50
phi = np.linspace(0, 2*np.pi, 1000)
WC2 = (np.sin(phi))*((a-(np.cos(phi)))**0.5)
plt.plot(phi,WC2,'r--')
plt.show()





-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# setting the axes at the centre
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.plot(x,y, 'r')
plt.show()


plt.plot(gani,I1, 'k', label='')
plt.plot(x,2*y, 'c', label='y=2sin(x)')
plt.plot(x,3*y, 'r', label='y=3sin(x)')

plt.legend(loc='upper left')

plt.show()





plt.xlabel('Age')
plt.ylabel('$\Delta T$')
plt.grid()
plt.show(block=False)
