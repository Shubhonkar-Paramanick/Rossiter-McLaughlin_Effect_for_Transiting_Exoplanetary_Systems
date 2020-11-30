#!/usr/bin/env python3
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from __future__ import division
from numpy import sqrt, sin, cos, pi, exp, arange
from numpy import arcsin as asin
from numpy import arccos as acos
from PyAstronomy.funcFit import OneDFit

plt.rcParams['figure.figsize'] = [12,7]

class rm(OneDFit):

    def __init__(self):
        OneDFit.__init__(self, ["xi", "zhe", "P", "Mid_Transit_Time", "i", "zeta", "Omega_s", "lambda", "a"])

    def planetDistance(self, theta):
        return self["a"]

    def W1(self, gani):
        result = ((2-(4*gani**2)+(3*gani**4)-(gani**6))/(2*((1-(gani**2))**1.5)))+(((self["zhe"]**2)*(-(2*gani**8)+(7*gani**6)-(10*gani**4)+(7*gani**2)-2))/(8*((1-(gani**2))**3.5)))+(((self["zhe"]**4)*((gani**8)-(10*gani**6)+(9*gani**4)+(8*gani**2)+8))/(192*((1-(gani**2))**5.5)))
        return result

    def W2(self, gani):
        result = sqrt(1.0 - gani**2) - self["zhe"]**2 * (4.0 - 3.0 * gani**2) / (8.0 * (1.0 - gani**2)**(3.0 / 2.0))
        return result

    def XpVec(self, theta):
        result = np.zeros(3)
        result[0] = -cos(self["lambda"]) * sin(theta) - sin(self["lambda"]) * cos(self["i"]) * cos(theta)
        result[1] = sin(self["i"]) * cos(theta)
        result[2] = sin(self["lambda"] * sin(theta)) - cos(self["lambda"]) * cos(self["i"]) * cos(theta)
        result *= self.rp(theta)
        return result

    def Xp(self, theta):
        result = -cos(self["lambda"]) * sin(theta) - sin(self["lambda"]) * cos(self["i"]) * cos(theta)
        result *= self.planetDistance(theta)
        return result

    def Zp(self, theta):
        result = sin(self["lambda"]) * sin(theta) - cos(self["lambda"]) * cos(self["i"]) * cos(theta)
        result *= self.planetDistance(theta)
        return result

    def etap(self, Xp, Zp):
        return sqrt(Xp**2 + Zp**2) - 1.0

    def VA_B(self, etap):
        result = (2.0 * etap + self["zhe"]**2 + etap**2) / (2.0 * (1.0 + etap))
        return result

    def ganiFromVec(self, XpVec):
        return sqrt(XpVec[0]**2 + XpVec[2]**2)

    def gani(self, Xp, Zp):
        return sqrt(Xp**2 + Zp**2)

    def True_Anomaly(self, time):
        result = ((time - self["Mid_Transit_Time"]) / self["P"] - np.floor((time - self["Mid_Transit_Time"]) / self["P"])) * 2.0 * np.pi
        return result

    def z0(self, etap, indi):
        result = np.zeros(etap.size)
        result[indi] = sqrt((self["zhe"]**2 - etap[indi]**2) * ((etap[indi] + 2.0)**2 - self["zhe"]**2)) / (2.0 * (1.0 + etap[indi]))
        return result

    def x0(self, etap):
        return 1.0 - (self["zhe"]**2 - etap**2) / (2.0 * (1.0 + etap))

    def g(self, x, e, g, x0):
        result = (1.0 - x**2) * asin(sqrt((g**2 - (x - 1.0 - e)**2) / (1.0 - x**2))) + sqrt(2.0 * (1.0 + e) * (x0 - x) * (g**2 - (x - 1.0 - e)**2))
        return result

    def xc(self, VA_B, x0):
        return x0 + (VA_B - self["zhe"]) / 2.0

    def W3(self, x0, VA_B, xc, etap):
        result = pi / 6.0 * (1.0 - x0)**2 * (2.0 + x0) + pi / 2.0 * self["zhe"] * (self["zhe"] - VA_B) * self.g(xc, etap, self["zhe"], x0) / self.g((1.0 - self["zhe"]), -self["zhe"], self["zhe"], x0) * self.W1(1.0 - self["zhe"])
        return result

    def W4(self, x0, VA_B, xc, etap):
        result = pi / 8. * (1.0 - x0)**2 * (1.0 + x0)**2 + pi / 2. * self["zhe"] * (self["zhe"] - VA_B) * xc * self.g(xc, etap, self["zhe"], x0) / self.g((1.0 - self["zhe"]), -self["zhe"], self["zhe"], x0) * self.W2(1.0 - self["zhe"])
        return result

    def evaluate(self, xOrig):

        x = self.True_Anomaly(xOrig)
        Xp = self.Xp(x)
        Zp = self.Zp(x)
        gani = self.gani(Xp, Zp)
        etap = self.etap(Xp, Zp)
        VA_B = self.VA_B(etap)
        x0 = self.x0(etap)
        xc = self.xc(VA_B, x0)

        PD = np.abs((xOrig - self["Mid_Transit_Time"]) / self["P"])
        PD = np.minimum(PD - np.floor(PD),
                               np.abs(PD - np.floor(PD) - 1))

        y = np.zeros(len(x))
        indi = np.where(np.logical_and(
            gani < (1.0 - self["zhe"]), PD < 0.25))[0]

        y[indi] = Xp[indi] * self["Omega_s"] * sin(self["zeta"]) * self["zhe"]**2 * (1.0 - self["xi"] * (1.0 - self.W2(gani[indi]))) / (1.0 - self["zhe"]**2 - self["xi"] * (1. /3. - self["zhe"]**2 * (1.0 - self.W1(gani[indi]))))

        indi = np.where(np.logical_and(
            np.logical_and(gani >= 1. - self["zhe"], gani < 1.0 + self["zhe"]), PD < 0.25))[0]
        z0 = self.z0(etap, indi)

        y[indi] = (Xp[indi] * self["Omega_s"] * sin(self["zeta"]) * (
            (1.0 - self["xi"]) * (-z0[indi] * VA_B[indi] + self["zhe"]**2 * acos(VA_B[indi] / self["zhe"])) +
            (self["xi"] / (1.0 + etap[indi])) * self.W4(x0[indi], VA_B[indi], xc[indi], etap[indi]))) / (pi * (1. - 1.0 / 3.0 * self["xi"]) - (1.0 - self["xi"]) * (asin(z0[indi]) - (1. + etap[indi]) * z0[indi] + self["zhe"]**2 * acos(VA_B[indi] / self["zhe"])) - self["xi"] * self.W3(x0[indi], VA_B[indi], xc[indi], etap[indi]))

        return y


# For HD209458 System
Rs = 1.203*(696340*((10)**3))
Omega_s = 2.0632008263473525e-05
V_s = Rs*Omega_s
RM = rm()
# Parameters Input

Angles = np.linspace(0,360,3)
A_size = len(Angles)

#for j in Angles:
    #time = np.linspace(-0.5,0.5,1000)
    #RM.assignValue({"a": 3.14, "lambda": j/180.0*pi, "xi": 0.1,"P":3.52474541, "Mid_Transit_Time": 0., "i": 87.8/180.*pi,"zeta": 90.0/180.0*pi, "Omega_s": 2.0632008263473525e-05, "zhe": 0.1})
    #RVA = Rs*RM.evaluate(time)
    #Rad_Ano[] = RVA
    #plt.plot(time, RVA, label='%s data' % j)



time = np.linspace(-0.5,0.5,1000)
RM.assignValue({"a": 3.14, "lambda": 0./180.0*pi, "xi": 0.1,"P":3.52474541, "Mid_Transit_Time": 0., "i": 87.8/180.*pi,"zeta": 90.0/180.0*pi, "Omega_s": 2.0632008263473525e-05, "zhe": 0.1})
RVA_1 = Rs*RM.evaluate(time)
plt.plot(time, RVA_1,color='#00CC66', label="$\mathrm{\lambda = 0^o}$")


RM.assignValue({"a": 3.14, "lambda": 180./180.0*pi, "xi": 0.1,"P":3.52474541, "Mid_Transit_Time": 0., "i": 87.8/180.*pi,"zeta": 90.0/180.0*pi, "Omega_s": 2.0632008263473525e-05, "zhe": 0.1})
RVA_2 = Rs*RM.evaluate(time)
plt.plot(time, RVA_2,color='#fcb568', label="$\mathrm{\lambda = 180^o}$")


plt.ylabel("Apparent Anomaly in the the Radial Velocity (m/s)", fontsize=14)
plt.xlabel("Time (in Days)", fontsize=14)
plt.legend(fontsize=14)
plt.grid()
plt.show()



###################################################Plots####################################################



time = np.linspace(-0.5,0.5,1000)


RM.assignValue({"a": 3.14, "lambda": -10./180.0*pi, "xi": 0.,"P":3.52474541, "Mid_Transit_Time": 0., "i": 87.8/180.*pi,"zeta": 3.3169335373771873/180.0*pi, "Omega_s": 2.0632008263473525e-05, "zhe": 0.1})
RVA_1 = Rs*RM.evaluate(time)
plt.plot(time, RVA_1,'k--', label="$\mathrm{\lambda = -10^o, V sin\zeta = 1 km/s}$")


RM.assignValue({"a": 3.14, "lambda": 0./180.0*pi, "xi": 0.,"P":3.52474541, "Mid_Transit_Time": 0., "i": 87.8/180.*pi,"zeta": 3.3169335373771873/180.0*pi, "Omega_s": 2.0632008263473525e-05, "zhe": 0.1})
RVA_2 = Rs*RM.evaluate(time)
plt.plot(time, RVA_2,'k-', label="$\mathrm{\lambda = 0^o, V sin\zeta = 1 km/s}$")


RM.assignValue({"a": 3.14, "lambda": 10./180.0*pi, "xi": 0.,"P":3.52474541, "Mid_Transit_Time": 0., "i": 87.8/180.*pi,"zeta": 3.3169335373771873/180.0*pi, "Omega_s": 2.0632008263473525e-05, "zhe": 0.1})
RVA_3 = Rs*RM.evaluate(time)
plt.plot(time, RVA_3,'k-.', label="$\mathrm{\lambda = 10^o, V sin\zeta = 1 km/s}$")


RM.assignValue({"a": 3.14, "lambda": -10./180.0*pi, "xi": 0.,"P":3.52474541, "Mid_Transit_Time": 0., "i": 87.8/180.*pi,"zeta": 6.6450491806133085/180.0*pi, "Omega_s": 2.0632008263473525e-05, "zhe": 0.1})
RVA_1 = Rs*RM.evaluate(time)
plt.plot(time, RVA_1,'r--', label="$\mathrm{\lambda = -10^o, V sin\zeta = 2 km/s}$")


RM.assignValue({"a": 3.14, "lambda": 0./180.0*pi, "xi": 0.,"P":3.52474541, "Mid_Transit_Time": 0., "i": 87.8/180.*pi,"zeta": 6.6450491806133085/180.0*pi, "Omega_s": 2.0632008263473525e-05, "zhe": 0.1})
RVA_2 = Rs*RM.evaluate(time)
plt.plot(time, RVA_2,'r-', label="$\mathrm{\lambda = 0^o, V sin\zeta = 2 km/s}$")


RM.assignValue({"a": 3.14, "lambda": 10./180.0*pi, "xi": 0.,"P":3.52474541, "Mid_Transit_Time": 0., "i": 87.8/180.*pi,"zeta": 6.6450491806133085/180.0*pi, "Omega_s": 2.0632008263473525e-05, "zhe": 0.1})
RVA_3 = Rs*RM.evaluate(time)
plt.plot(time, RVA_3,'r-.', label="$\mathrm{\lambda = 10^o, V sin\zeta = 2 km/s}$")


RM.assignValue({"a": 3.14, "lambda": -10./180.0*pi, "xi": 0.,"P":3.52474541, "Mid_Transit_Time": 0., "i": 87.8/180.*pi,"zeta": 9.995872958533576/180.0*pi, "Omega_s": 2.0632008263473525e-05, "zhe": 0.1})
RVA_1 = Rs*RM.evaluate(time)
plt.plot(time, RVA_1,'g--', label="$\mathrm{\lambda = -10^o, V sin\zeta = 3 km/s}$")


RM.assignValue({"a": 3.14, "lambda": 0./180.0*pi, "xi": 0.,"P":3.52474541, "Mid_Transit_Time": 0., "i": 87.8/180.*pi,"zeta": 9.995872958533576/180.0*pi, "Omega_s": 2.0632008263473525e-05, "zhe": 0.1})
RVA_2 = Rs*RM.evaluate(time)
plt.plot(time, RVA_2,'g-', label="$\mathrm{\lambda = 0^o, V sin\zeta = 3 km/s}$")


RM.assignValue({"a": 3.14, "lambda": 10./180.0*pi, "xi": 0.,"P":3.52474541, "Mid_Transit_Time": 0., "i": 87.8/180.*pi,"zeta": 9.995872958533576/180.0*pi, "Omega_s": 2.0632008263473525e-05, "zhe": 0.1})
RVA_3 = Rs*RM.evaluate(time)
plt.plot(time, RVA_3,'g-.', label="$\mathrm{\lambda = 10^o, V sin\zeta = 3 km/s}$")


RM.assignValue({"a": 3.14, "lambda": -10./180.0*pi, "xi": 0.,"P":3.52474541, "Mid_Transit_Time": 0., "i": 87.8/180.*pi,"zeta": 13.381648745718064/180.0*pi, "Omega_s": 2.0632008263473525e-05, "zhe": 0.1})
RVA_1 = Rs*RM.evaluate(time)
plt.plot(time, RVA_1,'b--', label="$\mathrm{\lambda = -10^o, V sin\zeta = 4 km/s}$")


RM.assignValue({"a": 3.14, "lambda": 0./180.0*pi, "xi": 0.,"P":3.52474541, "Mid_Transit_Time": 0., "i": 87.8/180.*pi,"zeta": 13.381648745718064/180.0*pi, "Omega_s": 2.0632008263473525e-05, "zhe": 0.1})
RVA_2 = Rs*RM.evaluate(time)
plt.plot(time, RVA_2,'b-', label="$\mathrm{\lambda = 0^o, V sin\zeta = 4 km/s}$")


RM.assignValue({"a": 3.14, "lambda": 10./180.0*pi, "xi": 0.,"P":3.52474541, "Mid_Transit_Time": 0., "i": 87.8/180.*pi,"zeta": 13.381648745718064/180.0*pi, "Omega_s": 2.0632008263473525e-05, "zhe": 0.1})
RVA_3 = Rs*RM.evaluate(time)
plt.plot(time, RVA_3,'b-.', label="$\mathrm{\lambda = 10^o, V sin\zeta = 4 km/s}$")


RM.assignValue({"a": 3.14, "lambda": -10./180.0*pi, "xi": 0.,"P":3.52474541, "Mid_Transit_Time": 0., "i": 87.8/180.*pi,"zeta": 16.815777461739835/180.0*pi, "Omega_s": 2.0632008263473525e-05, "zhe": 0.1})
RVA_1 = Rs*RM.evaluate(time)
plt.plot(time, RVA_1,'y--', label="$\mathrm{\lambda = -10^o, V sin\zeta = 5 km/s}$")


RM.assignValue({"a": 3.14, "lambda": 0./180.0*pi, "xi": 0.,"P":3.52474541, "Mid_Transit_Time": 0., "i": 87.8/180.*pi,"zeta": 16.815777461739835/180.0*pi, "Omega_s": 2.0632008263473525e-05, "zhe": 0.1})
RVA_2 = Rs*RM.evaluate(time)
plt.plot(time, RVA_2,'y-', label="$\mathrm{\lambda = 0^o, V sin\zeta = 5 km/s}$")


RM.assignValue({"a": 3.14, "lambda": 10./180.0*pi, "xi": 0.,"P":3.52474541, "Mid_Transit_Time": 0., "i": 87.8/180.*pi,"zeta": 16.815777461739835/180.0*pi, "Omega_s": 2.0632008263473525e-05, "zhe": 0.1})
RVA_3 = Rs*RM.evaluate(time)
plt.plot(time, RVA_3,'y-.', label="$\mathrm{\lambda = 10^o, V sin\zeta = 5 km/s}$")



plt.title("Apparent Anomaly in the the Radial Velocity with linear limb darkening", fontsize=17)
plt.ylabel("Radial Velocity Anomaly (m/s)", fontsize=14)
plt.xlabel("Time (in Days)", fontsize=14)
plt.legend(fontsize=8)
plt.grid()
plt.show()



#######################################################################################################

# Change in anomaly due to the variation in parameters


#######################################################################################################

#Stellar Radius


Rs = 1.203*(696340*((10)**3))
R_s = Rs-(10**(7))
    
n = 20    
for k in range(0, n):
    k=k+(10**(7))
    time = np.linspace(-0.5,0.5,1000)
    RM.assignValue({"a": 3.14, "lambda": 0./180.0*pi, "xi": 0.1,"P":3.52474541, "Mid_Transit_Time": 0., "i": 87.8/180.*pi,"zeta": 90.0/180.0*pi, "Omega_s": 2.0632008263473525e-05, "zhe": 0.1})
    RVA = (R_s+k)*RM.evaluate(time)
    RVA_Var = k*RM.evaluate(time)
    Norm_RVA_Var = RVA_Var/((10**(7))/1.203*(696340*((10)**3)))
    plt.plot(time, Norm_RVA_Var,'r-', alpha =0.2)
    
    
plt.title("Variation in Apparent Anomaly in the the Radial Velocity \n with change in Stellar Radius (Limb Darkened Source)", fontsize=15)
plt.ylabel("Change in Radial Velocity Apparent Anomaly (m/s)", fontsize=12)
plt.xlabel("Time (in Days)", fontsize=12)
#plt.legend(fontsize=8)
plt.grid()
plt.show()


 


Rs = 1.203*(696340*((10)**3)) 
R_s = np.linspace(Rs-(10**(8)),Rs+(10**(8)),2000) 
for o in R_s: 
    time = np.linspace(-0.5,0.5,1000)
    RM.assignValue({"a": 3.14, "lambda": 0./180.0*pi, "xi": 0.1,"P":3.52474541, "Mid_Transit_Time": 0., "i": 87.8/180.*pi,"zeta": 90.0/180.0*pi, "Omega_s": 2.0632008263473525e-05, "zhe": 0.1})
    RVA = o*RM.evaluate(time)
    plt.plot(time, RVA,'g-.', alpha =0.2)



plt.title("Apparent Anomaly in the the Radial Velocity \n with change in Stellar Radius (Limb Darkened Source, Step size: 1 ppm)", fontsize=15)
plt.ylabel("Apparent Anomaly in the the Radial Velocity (m/s)", fontsize=12)
plt.xlabel("Time (in Days)", fontsize=12)
#plt.legend(fontsize=8)
plt.grid()
plt.show()



