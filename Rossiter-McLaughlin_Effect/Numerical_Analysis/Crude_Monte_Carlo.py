#!/usr/bin/env python3
import numpy as np
import scipy as sp
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm
from tqdm import tqdm_gui
from alive_progress import alive_bar
from mpl_toolkits.mplot3d import Axes3D

----------------------------------------------------------------------------------------------------------------------
#Crude Monte Carlo

def random_number(minimum_number, maximum_number):
    random_in_range = (maximum_number-minimum_number)*(random.uniform(0,1))
    return minimum_number + random_in_range
'''    
a = np.zeros((500))
for i in range(0,500):
    a[i] = random_number(6,10)
    
plt.plot(a,'r.')
plt.show()
'''

gamma = 0.3
rho = np.linspace(0, 1-gamma, 1000)

#WC = (-gamma**2)*(1/(4*((1-(rho**2))**0.5)))


def f_of_x(phi):
    return ((gamma)*(np.cos(phi))/(np.pi*rho**1))*(((1/(24*gamma**4))*(((10*gamma*rho*np.cos(phi))*((1-(rho**2)-(gamma**2)-(2*gamma*rho*np.cos(phi)))**1.5))-((6*gamma**2)*((1-(rho**2)-(gamma**2)-(2*gamma*rho*np.cos(phi)))**1.5))-((3*gamma)*((gamma)+(rho*np.cos(phi)))*((rho**2)-1-(5*((rho*np.cos(phi))**2)))*((1-(rho**2)-(gamma**2)-(2*gamma*rho*np.cos(phi)))**0.5))-((3*gamma/2)*((rho**2)-1-(5*((rho*np.cos(phi))**2)))*(2+((rho**2)*(np.cos(2*phi)))-(rho**2))*(np.arctan(((gamma)+(rho*np.cos(phi)))/((1-(rho**2)-(gamma**2)-(2*gamma*rho*np.cos(phi)))**0.5))))))-((1/(24*gamma**4))*(((10*gamma*rho*np.cos(phi))*((1-(rho**2))**1.5))-((3*gamma*rho*np.cos(phi))*((1-(rho**2))**0.5)*((rho**2)-1-(5*((rho*np.cos(phi))**2))))-((3*gamma/2)*((rho**2)-1-(5*((rho*np.cos(phi))**2)))*(2+((rho**2)*(np.cos(2*phi)))-(rho**2))*(np.arctan(((rho*np.cos(phi)))/((1-(rho**2))**0.5)))))))
    
    
    #((rho/(2*np.pi*gamma**2))*(((np.cos(phi))**3)*(1-((rho*np.sin(phi))**2))*((np.arcsin(((gamma)+(rho*np.cos(phi)))/((1-((rho*np.sin(phi))**2))**0.5)))-(np.arcsin(((rho*np.cos(phi)))/((1-((rho*np.sin(phi))**2))**0.5)))))) + ((1/(8*rho*np.pi*gamma**2))*(((np.cos(phi))**1)*((1-((rho*np.sin(phi))**2))**2)*((np.arcsin(((gamma)+(rho*np.cos(phi)))/((1-((rho*np.sin(phi))**2))**0.5)))-(np.arcsin(((rho*np.cos(phi)))/((1-((rho*np.sin(phi))**2))**0.5)))))) + (((5*rho**2)/(8*np.pi*gamma**2))*(((np.cos(phi))**4)*((1-(rho**2)-(gamma**2)-(2*gamma*rho*np.cos(phi)))**0.5))) + (((7*rho**1)/(8*np.pi*gamma))*(((np.cos(phi))**3)*((1-(rho**2)-(gamma**2)-(2*gamma*rho*np.cos(phi)))**0.5))) + (((3)/(8*np.pi))*(((np.cos(phi))**2)*((1-(rho**2)-(gamma**2)-(2*gamma*rho*np.cos(phi)))**0.5))) + (((gamma)/(8*np.pi*rho))*(((np.cos(phi))**1)*((1-(rho**2)-(gamma**2)-(2*gamma*rho*np.cos(phi)))**0.5))) - (((1)/(8*np.pi*gamma*rho))*(((np.cos(phi))**1)*((1-(rho**2)-(gamma**2)-(2*gamma*rho*np.cos(phi)))**1.5))) + (((13)/(24*np.pi*gamma**2))*(((np.cos(phi))**2)*((1-(rho**2)-(gamma**2)-(2*gamma*rho*np.cos(phi)))**1.5))) - ((13/(24*gamma**2))*((1-(rho**2))**1.5)) - ((15/(32*gamma**2))*((1-(rho**2))**0.5)*(rho**2))
    
    
    #return ((-1/(2*np.pi*gamma**2))*(((np.cos(phi))**1)*((gamma)+(rho*np.cos(phi)))*(rho)*((1-(rho**2)-(gamma**2)-(2*gamma*rho*np.cos(phi)))**0.5))) - ((1/(3*np.pi*gamma**2))*((1-(rho**2)-(gamma**2)-(2*gamma*rho*np.cos(phi)))**1.5)) - ((1/(2*np.pi*gamma**2))*(((np.cos(phi))**1)*(1-((rho*np.sin(phi))**2))*((np.arcsin(((gamma)+(rho*np.cos(phi)))/((1-((rho*np.sin(phi))**2))**0.5)))-(np.arcsin(((rho*np.cos(phi)))/((1-((rho*np.sin(phi))**2))**0.5)))))) + ((2/(3*gamma**2))*((1-(rho**2))**0.5)*(1-(0.25*rho**2)))
    
    
    #((rho/(2*np.pi*gamma**2))*(((np.cos(phi))**3)*(1-((rho*np.sin(phi))**2))*((np.arcsin(((gamma)+(rho*np.cos(phi)))/((1-((rho*np.sin(phi))**2))**0.5)))-(np.arcsin(((rho*np.cos(phi)))/((1-((rho*np.sin(phi))**2))**0.5)))))) + ((1/(8*rho*np.pi*gamma**2))*(((np.cos(phi))**1)*((1-((rho*np.sin(phi))**2))**2)*((np.arcsin(((gamma)+(rho*np.cos(phi)))/((1-((rho*np.sin(phi))**2))**0.5)))-(np.arcsin(((rho*np.cos(phi)))/((1-((rho*np.sin(phi))**2))**0.5)))))) + (((5*rho**2)/(8*np.pi*gamma**2))*(((np.cos(phi))**4)*((1-(rho**2)-(gamma**2)-(2*gamma*rho*np.cos(phi)))**0.5))) + (((7*rho**1)/(8*np.pi*gamma))*(((np.cos(phi))**3)*((1-(rho**2)-(gamma**2)-(2*gamma*rho*np.cos(phi)))**0.5))) + (((3)/(8*np.pi))*(((np.cos(phi))**2)*((1-(rho**2)-(gamma**2)-(2*gamma*rho*np.cos(phi)))**0.5))) + (((gamma)/(8*np.pi*rho))*(((np.cos(phi))**1)*((1-(rho**2)-(gamma**2)-(2*gamma*rho*np.cos(phi)))**0.5))) - (((1)/(8*np.pi*gamma*rho))*(((np.cos(phi))**1)*((1-(rho**2)-(gamma**2)-(2*gamma*rho*np.cos(phi)))**1.5))) + (((13)/(24*np.pi*gamma**2))*(((np.cos(phi))**2)*((1-(rho**2)-(gamma**2)-(2*gamma*rho*np.cos(phi)))**1.5))) - ((13/(24*gamma**2))*((1-(rho**2))**1.5)) - ((15/(32*gamma**2))*((1-(rho**2))**0.5)*(rho**2)) - ((1/(2*np.pi*gamma**2))*(((np.cos(phi))**1)*((gamma)+(rho*np.cos(phi)))*(rho)*((1-(rho**2)-(gamma**2)-(2*gamma*rho*np.cos(phi)))**0.5))) - ((1/(3*np.pi*gamma**2))*((1-(rho**2)-(gamma**2)-(2*gamma*rho*np.cos(phi)))**1.5)) - ((1/(2*np.pi*gamma**2))*(((np.cos(phi))**1)*(1-((rho*np.sin(phi))**2))*((np.arcsin(((gamma)+(rho*np.cos(phi)))/((1-((rho*np.sin(phi))**2))**0.5)))-(np.arcsin(((rho*np.cos(phi)))/((1-((rho*np.sin(phi))**2))**0.5)))))) + ((2/(3*gamma**2))*((1-(rho**2))**0.5)*(1-(0.25*rho**2)))    
    
    
    
def crude_monte_carlo(num_samples=5000):
    """
    This function performs the Crude Monte Carlo for our
    specific function f(x) on the range x=0 to x=5.
    Notice that this bound is sufficient because f(x)
    approaches 0 at around PI.
    Args:
    - num_samples (float) : number of samples
    Return:
    - Crude Monte Carlo estimation (float)
    """
    lower_bound = 0
    upper_bound = 2*np.pi
    c = np.zeros((1000,num_samples))
    rho = np.array([np.linspace(0, 1-gamma, 1000)]).T
    delete1 = np.zeros(num_samples)
    sum_of_samples = 0
    for i in range(num_samples):
        phi = random_number(lower_bound, upper_bound)
        delete1[i] = phi
        #print(phi)       
        samples = f_of_x(phi)
        c[:,i] = samples
    sum_ = np.array([np.sum(c,axis=1)]).T
    Crude_MC = (upper_bound-lower_bound)*(sum_/num_samples)
    return {'Crude_MC':Crude_MC, 'c':c ,'sum_':sum_ }

MC_samples = 10000
crude_estimation = crude_monte_carlo(MC_samples)
Crude_MC = np.array(list(crude_estimation.items())[0], dtype=dict)[1]
c = np.array(list(crude_estimation.items())[1], dtype=dict)[1]
sum_ = np.array(list(crude_estimation.items())[2], dtype=dict)[1]


# METHOD-1    
    
def get_crude_MC_variance(C=c,Function_Sum=sum_,num_samples=MC_samples):
    """
    This function returns the variance for the Crude Monte Carlo.
    Note that the inputed number of samples does not neccissarily
    need to correspond to number of samples used in the Monte
    Carlo Simulation.
    Args:
    - num_samples (int)
    Return:
    - Variance for Crude Monte Carlo approximation of f(x) (float)
    """
    lower_bound = 0
    upper_bound = 2*np.pi
    c_sq = C**2
    c_sq_sum = np.array([np.sum(c_sq,axis=1)]).T
    sum_of_sqs = ((upper_bound-lower_bound)**2)*c_sq_sum/num_samples
    # get square of average

    sq_ave = ((upper_bound-lower_bound)*Function_Sum/num_samples)**2
    
    return sum_of_sqs - sq_ave
    
    
# Now we will run a Crude Monte Carlo simulation with 10000 samples
# We will also calculate the variance with 10000 samples and the error


Integration_variance = (get_crude_MC_variance(c,sum_,MC_samples))/MC_samples
Integration_error = (Integration_variance)**0.5

plt.plot(rho,Crude_MC,'k--')
plt.plot(rho,Integration_variance,'g--')
plt.plot(rho,Integration_error,'r--')
plt.show()





# METHOD-2

def function_variance(C=c,Function_Sum=sum_,num_samples=MC_samples):

    c_sq = C**2
    c_sq_sum = np.array([np.sum(c_sq,axis=1)]).T
    avg_of_sq = c_sq_sum/num_samples
    # get square of average

    sq_of_ave = (Function_Sum/num_samples)**2
    
    return avg_of_sq-sq_of_ave


func_variance = function_variance(c,sum_,MC_samples)
func_error = (func_variance/MC_samples)**0.5



MC_samples = 10000
MC_samples2 = 250
Crude_MC_repeat = np.zeros((1000,MC_samples2))
#c_repeat = np.zeros((1000,MC_samples,MC_samples2), dtype='float64')
#sum_repeat = np.zeros((1000,MC_samples2))
'''
If Memory Error is displayed, change the overcommit mode to 1 by running the following command in terminal:
echo 1 | sudo tee /proc/sys/vm/overcommit_memory

Also, for c_repeat, the dtype can be changed to uint8.
'''


with alive_bar(MC_samples2,title='Computing...',length=20) as bar:
    for j in range(MC_samples2):
        crude_estimation = crude_monte_carlo(MC_samples)
        Crude_MC_repeat[:,j] = np.array(list(crude_estimation.items())[0], dtype=dict)[1].reshape((1000,))
        plt.plot(rho,Crude_MC_repeat[:,j], 'c', alpha=0.1)
        #c_repeat[:,:,j] = np.array(list(crude_estimation.items())[1], dtype=dict)[1]
        #sum_repeat[:,j] = np.array(list(crude_estimation.items())[2], dtype=dict)[1].reshape((1000,))
        bar()
plt.show()



def integration_variance(CRUDE_MC_REPEAT=Crude_MC_repeat,num_samples=MC_samples2):

    Inte_Matrix_sq = Crude_MC_repeat**2

    
    Inte_Matrix_sq_sum = np.array([np.sum(Inte_Matrix_sq,axis=1)]).T
    Inte_avg_of_sq = Inte_Matrix_sq_sum/num_samples
    # get square of average

    Inte_Matrix_sum = np.array([np.sum(Crude_MC_repeat,axis=1)]).T
    Inte_sq_of_ave = (Inte_Matrix_sum/num_samples)**2
    
    return Inte_avg_of_sq-Inte_sq_of_ave
    


Inte_variance = integration_variance(Crude_MC_repeat,MC_samples2)
Inte_error = (Inte_variance)**0.5

plt.plot(rho,Inte_variance,'c--')
plt.plot(rho,Inte_error,'y--')
plt.show()







---------------------------------------------------------------------------------------------------------------
# Importance Sampling




rho = np.linspace(0, 1-gamma, 1000)
phi = np.linspace(0, 2*np.pi, 1000)
RHO, PHI = np.meshgrid(rho, phi)
GAMMA = gamma

Integrand1 = (((RHO**2)*(np.cos(PHI)))+(RHO*GAMMA))*(np.cos(PHI))*((1-(RHO**2)-(GAMMA**2)-(2*RHO*GAMMA*np.cos(PHI)))**0.5)

fig1 = plt.figure(1)
ax = Axes3D(fig1)
ax.plot_surface(RHO, PHI, Integrand1)
plt.xlabel('rho')
plt.ylabel('phi')
plt.show()
fig2 = plt.figure(2)
plt.contourf(RHO, PHI, Integrand1)
plt.colorbar()
plt.xlabel('rho')
plt.ylabel('phi')
plt.show()




rho = np.linspace(0, 1-gamma, 3)
phi = np.linspace(0, 2*np.pi, 1000)

x_data = np.zeros((len(phi),len(rho)))
y_data = np.zeros((len(phi),len(rho)))


for k in range(len(rho)):
    func_at_diff_rho = ((((rho[k])**2)*(np.cos(phi)))+((rho[k])*gamma))*(np.cos(phi))*((1-((rho[k])**2)-(gamma**2)-(2*(rho[k])*gamma*np.cos(phi)))**0.5)
    plt.plot(phi,func_at_diff_rho,'k--')
    x_data[:,k] = phi
    y_data[:,k] = func_at_diff_rho
plt.show()









sine_cos_sh = (np.sin(20*phi + 62.8))*(np.cos(phi + 3.14))
#sine_cos_sh2 = (np.sin(10*phi + 100))*(np.cos(phi + 5))
plt.plot(phi,sine_cos_sh)
#plt.plot(phi,sine_cos_sh2)     
plt.show()





----------------------------------------------------------------------------------------------------------------------


x_model = np.linspace(0, 2*np.pi, 1000)

#
# Radial Velocity Model
#
def importance_function(x,lambda1,lambda2,lambda3,lambda4,lambda5):
    return lambda1 + ((lambda2)*(np.sin(lambda3*(x+lambda4)))*(np.cos(x+lambda5)))

#
# Initial Parameters, Lower and Upper Bounds
#
Parameter0 = np.array([0.03, 0.2, 1.0, 0.0, 0.0])
lowerbnd  = np.array([-0.2, 0.0, -5.0, 0.0, 0.0])
upperbnd  = np.array([0.2, 0.5, 5.0, 2*np.pi, 2*np.pi])
bounds = (lowerbnd,upperbnd)
ndim = len(Parameter0)
#
# Least Squares Fit
#
f = curve_fit(importance_function, x_data[:,1], y_data[:,1], Parameter0, bounds=bounds)
lambda1_lsq, lambda2_lsq, lambda3_lsq, lambda4_lsq, lambda5_lsq = f[0]
best_fit_model = importance_function(x_model,lambda1_lsq, lambda2_lsq, lambda3_lsq, lambda4_lsq, lambda5_lsq)

plt.plot(x_model,best_fit_model,color='#87CEFA', label="LS Estimate (SciPy Optimize)")
plt.plot(x_data[:,1],y_data[:,1],color='k', label="")
plt.xlabel('')
plt.ylabel('')
plt.legend(fontsize=14)
plt.grid()
plt.show()




----------------------------------------------------------------------------------------------------------------------



data = (x_data[2],y_data[2])
#
# Model
#
def importance_function_ml(Parameters,x=x_model):
    lambda1,lambda2,lambda3,lambda4,lambda5 = Parameters
    return lambda1 + ((lambda2)*(np.sin(lambda3*(x+lambda4)))*(np.cos(x+lambda5)))
#
# Likelihood Function
#
def lnlike(Parameters,x,y):
    return -0.5*(np.sum((np.log(2*np.pi))+(((y-importance_function_ml(Parameters,x)))**2)))

#
# Initial Parameters, Lower and Upper Bounds
#
Param0 = np.array([0.03, 0.2, 1.0, 0.0, 0.0])
bounds = ((-0.2,0.2),(0.0,0.5),(-5.0,5.0),(0.0,2*np.pi),(0.0,2*np.pi))

#
# Maximum Likelihood Fit
#
ln = lambda*args: -lnlike(*args)
output = minimize(ln, Param0, args=data, bounds=bounds, tol= 1e-10)
lambda1_ml, lambda2_ml, lambda3_ml, lambda4_ml, lambda5_ml = output["x"]
ml_estimates = np.array([lambda1_ml, lambda2_ml, lambda3_ml, lambda4_ml, lambda5_ml])
ndim = len(ml_estimates)
best_fit_model_ml = importance_function_ml(ml_estimates,x_model)


plt.plot(x_model,best_fit_model_ml,color='#008080', label="Maximum Likelihood Fit")
plt.plot(x_data[:,2],y_data[:,2],color='k', label="")
plt.xlabel('')
plt.ylabel('')
plt.legend(fontsize=14)
plt.grid()
plt.show()




-----------------------------------------------------------------------------------------------------------------


# SciPy Basinhopping










































































# this is the template of our weight function g(x)
def g_of_x(x, A, lamda):
    e = 2.71828
    return A*math.pow(e, -1*lamda*x)
    
    
    
xs = [float(i/50) for i in range(int(50*PI))]
fs = [f_of_x(x) for x in xs]
gs = [g_of_x(x, A=1.4, lamda=1.4) for x in xs]
plt.plot(xs, fs)
plt.plot(xs, gs)
plt.title("f(x) and g(x)");


def inverse_G_of_r(r, lamda):
    return (-1 * math.log(float(r)))/lamda
    
    
def get_IS_variance(lamda, num_samples):
    """
    This function calculates the variance if a Monte Carlo
    using importance sampling.
    Args:
    - lamda (float) : lamdba value of g(x) being tested
    Return: 
    - Variance
    """
    A = lamda
    int_max = 5
    
    # get sum of squares
    running_total = 0
    for i in range(num_samples):
        x = random_number(0, int_max)
        running_total += (f_of_x(x)/g_of_x(x, A, lamda))**2
    
    sum_of_sqs = running_total / num_samples
    
    # get squared average
    running_total = 0
    for i in range(num_samples):
        x = random_number(0, int_max)
        running_total += f_of_x(x)/g_of_x(x, A, lamda)
    sq_ave = (running_total/num_samples)**2
    
    
    return sum_of_sqs - sq_ave
    
    
    
# get variance as a function of lambda by testing many
# different lambdas

test_lamdas = [i*0.05 for i in range(1, 61)]
variances = []

for i, lamda in enumerate(test_lamdas):
    print(f"lambda {i+1}/{len(test_lamdas)}: {lamda}")
    A = lamda
    variances.append(get_IS_variance(lamda, 10000))
    clear_output(wait=True)
    
optimal_lamda = test_lamdas[np.argmin(np.asarray(variances))]
IS_variance = variances[np.argmin(np.asarray(variances))]

print(f"Optimal Lambda: {optimal_lamda}")
print(f"Optimal Variance: {IS_variance}")
print((IS_variance/10000)**0.5)



plt.plot(test_lamdas[5:40], variances[5:40])
plt.title("Variance of MC at Different Lambda Values");



def importance_sampling_MC(lamda, num_samples):
    A = lamda
    
    running_total = 0
    for i in range(num_samples):
        r = random_number(0,1)
        running_total += f_of_x(inverse_G_of_r(r, lamda=lamda))/g_of_x(inverse_G_of_r(r, lamda=lamda), A, lamda)
    approximation = float(running_total/num_samples)
    return approximation
    
    
    
    
    
# run simulation
num_samples = 10000
approx = importance_sampling_MC(optimal_lamda, num_samples)
variance = get_IS_variance(optimal_lamda, num_samples)
error = (variance/num_samples)**0.5

# display results
print(f"Importance Sampling Approximation: {approx}")
print(f"Variance: {variance}")
print(f"Error: {error}")

