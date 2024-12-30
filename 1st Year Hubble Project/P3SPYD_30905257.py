#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:33:13 2020

@author: liutauras
"""
import math
import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as opt
import matplotlib.pyplot as pyplt

pyplt.rcParams["figure.figsize"] = (10, 10)

def chisq(M, sig_M, M_exp): #Function to find the Chi^squared value
    chi_2 = np.sum(((M-M_exp)**2.0)/(sig_M**2.0))
    return chi_2

def LP_Relation(gradient,x,intercept): #The linear function used throughout
    y= gradient*x + intercept 
    return y
parallax, err_par, period, m, A, err_A = np.loadtxt('MW_Cepheids.dat',\
                                 unpack=True, \
                                 usecols=(1,2,3,4,5,6), \
                                 dtype=float)

star= np.loadtxt('MW_Cepheids.dat',\
                                 unpack=True, \
                                 usecols=(0), \
                                 dtype=str)
#Processing the data
distance=1000/parallax
period_log=np.array([math.log(i,10) for i in period])
log_dist=np.array([math.log(i,10) for i in distance])
M=m-5*log_dist-A+5
#Degrees of freedom:
df=len(distance)-2
#Error propagation:
sig_d=((1000/(parallax**2))*err_par)#Not using -1000 because later on this value is squared
sig_logd=((5/(np.log(10)*distance))*sig_d) #Error in log_d
sig_M=np.sqrt(sig_logd**2+err_A**2) #Total error of Absolute Magnitude

#Using curve fit(the LM algorithm) method to find chi2
start_intercept = -10 
start_gradient= 10
opt_parameters, covar = opt.curve_fit(f=LP_Relation, xdata=period_log, ydata=M, sigma=sig_M,
                                        p0=(start_gradient, start_intercept),
                                       absolute_sigma=True)
#Analysing data from the LM algorithm
opt_gradient = opt_parameters[0]
opt_intercept = opt_parameters[1]
M_Pred=LP_Relation(opt_gradient,period_log,opt_intercept)
opt_chi2=chisq(M,sig_M,M_Pred)
err_gradient=np.sqrt(covar[0,0])
err_intercept=np.sqrt(covar[1,1])
correlation_m=([covar[0,0]/(err_gradient**2),covar[0,1]/(err_gradient*err_intercept)],
               [covar[1,0]/(err_gradient*err_intercept),covar[1,1]/(err_intercept**2)])

#I created this to find an acceptable value by which my x-values should be shifted
#to get a corellation as close as possible to 0 between alpha and beta. It basically
#keeps reiterating the curve_fit() function until the correlation coefficient is <=0.001.
a=0#This will be the amount of times the function reiterates
b=-0.3#this will be the value by which the x-values is shifted after the loop stops running
period_log_shifted=period_log-0.3
statement=False
while statement==False:
    a=a+1
    b=b-0.001
    period_log_shifted=period_log_shifted-0.001 #Shifting the x values slowly
    #Using curve fit(the LM algorithm) method to find shifted chi2
    start_intercept = -10 
    start_gradient= 10
    opt_parameters_shifted, covar_shifted = opt.curve_fit(f=LP_Relation, xdata=period_log_shifted, ydata=M, sigma=sig_M,
                                                          p0=(start_gradient, start_intercept),
                                                          absolute_sigma=True)
    err_gradient_shifted=np.sqrt(covar_shifted[0,0])
    err_intercept_shifted=np.sqrt(covar_shifted[1,1])
    value=covar_shifted[1,0]/(err_gradient_shifted*err_intercept_shifted)
    if np.absolute(value)<=0.001:
        statement=True
        opt_gradient_shifted = opt_parameters_shifted[0]
        opt_intercept_shifted = opt_parameters_shifted[1]
        M_Pred=LP_Relation(opt_gradient_shifted,period_log_shifted,opt_intercept_shifted)
        opt_chi2_shifted=chisq(M,sig_M,M_Pred)

correlation_m_shifted=([covar_shifted[0,0]/(err_gradient_shifted**2),covar_shifted[0,1]/(err_gradient_shifted*err_intercept_shifted)],
               [covar_shifted[1,0]/(err_gradient_shifted*err_intercept_shifted),covar_shifted[1,1]/(err_intercept_shifted**2)])

#Plotting the data(Values plotted are the shifted values are those are the ones with minimal corellation
#between alpha and beta)
plt.clf()
plt.rcParams['font.family']='cursive'
plt.errorbar(period_log_shifted,M,yerr=sig_M,capsize=4,ls='none',elinewidth=0.5,ecolor='red')
plt.scatter(period_log_shifted,M,s=20,color='black',marker='x')
#Plotting the predicted model with the shifted x values:
x=np.linspace(-0.35,0.8,100)
plt.plot(x,LP_Relation(opt_gradient_shifted,x,opt_intercept_shifted),color='black',linewidth=0.5)
plt.xlabel('Period log(P) ', color='black',size=10)
plt.ylabel('Absolute Magnitude M',color='black',size=10)
plt.title('Period-Luminosity relation for Cepheids(with shifted x-values)', color='black',size=15)
#Plotting the text(alpha):
plt.text(-0.2,-5.0,'Optimum Alpha:',color='black',fontsize=7)
plt.text(0,-5,np.round(opt_gradient_shifted,4),color='red',fontsize=10)
plt.text(0.15,-5,s='+/-',color='black',fontsize=7)
plt.text(0.2,-5,np.round(err_gradient,5),color='black',fontsize=7)
#Plotting the text(beta):
plt.text(-0.2,-4.85,'Optimum Beta',color='black',fontsize=7)
plt.text(0,-4.85,s=np.round(opt_intercept_shifted,4),color='red',fontsize=10)
plt.text(0.15,-4.85,s='+/-',color='black',fontsize=7)
plt.text(0.2,-4.85,np.round(err_intercept,5),color='black',fontsize=7)
#Plotting the text(red chi):
plt.text(-0.2,-4.7,s='Red Chi^2:',color='black',fontsize=7)
plt.text(-0.05,-4.7,np.round(opt_chi2_shifted/df,4),color='red',fontsize=10)
#Plotting the star names
for i in range(10):
    c=period_log_shifted[i]+0.01
    d=M[i]
    plt.annotate(star[i],xy=(c,d),fontsize=9)

plt.gca().invert_yaxis() #inverting the y-axis to make it look nicer
plt.show()
#Where im putting all the results from the data
print('===============================================')
print('Levenberg-Marquardt algorithm RESULTS')
print('===============================================')
print('Non-Shifted results')
print('===============================================')
print('Corellation matrix:')
print(correlation_m)
print('Best-fitting Alpha value(gradient)  = ', opt_gradient)
print('Best-fitting Beta value(intercept) = ',opt_intercept)
print('Corresponding minimum Chi^2 = ', opt_chi2)
print('Corresponding minimum Reduced Chi^2 =', opt_chi2/df)
print("Error on Alpha(gradient):", err_gradient)
print("Error on Beta(intercept):", err_intercept)
print('===============================================')
print('Shifted x-axis results')
print('===============================================')
print('Shifted-Corellation matrix: (Found after',a,'attempts) The "shifting" value for the x axis is:',np.round(b,5))
print(correlation_m_shifted)
print('Best-fitting Alpha value(gradient)  = ', opt_gradient_shifted)
print('Best-fitting Beta value(intercept) = ',opt_intercept_shifted)
print('Corresponding minimum Chi^2 = ', opt_chi2_shifted)
print('Corresponding minimum Reduced Chi^2 =', opt_chi2_shifted/df)
print("Error on Alpha(gradient):", err_gradient_shifted)
print("Error on Beta(intercept) :", err_intercept_shifted)
print('===============================================')
alpha=opt_gradient_shifted
beta=opt_intercept_shifted
err_alpha=err_gradient_shifted
err_beta=err_intercept_shifted
x_shift=b
print()
print('STEP 2:')
print()
# =============================================================================
# This is step 2
# =============================================================================
rvel, err_A = np.loadtxt('galaxy_data.dat',\
                                 unpack=True, \
                                 usecols=(1,2), \
                                 dtype=float)
galaxy= np.loadtxt('galaxy_data.dat',\
                                 unpack=True, \
                                 usecols=(0), \
                                 dtype=str)

# These are the lists that will include a list from each of the 8 galaxies
list_cepheid=[[],[],[],[],[],[],[],[]]
list_log=[[],[],[],[],[],[],[],[]]
list_magnitudes=[[],[],[],[],[],[],[],[]]
list_err_d=[[],[],[],[],[],[],[],[]]
list_chi2=[[],[],[],[],[],[],[],[]]
for i in range(8):
    cepheids=np.loadtxt('hst_gal%d_cepheids.dat'%(i+1),#Using i+1 so it reads from 1-8 galaxies not 0-7
                     unpack=True, \
                     usecols=(0), \
                     dtype=str,)
    list_cepheid[i]=cepheids
    logP, m =np.loadtxt('hst_gal%d_cepheids.dat'%(i+1),\
                     unpack=True, \
                     usecols=(1,2), \
                     dtype=float)
    list_log[i]=logP
    list_magnitudes[i]=m

list_logP=np.array(list_log) #Converting into numpy arrays so I can manipulate these lists
list_m=np.array(list_magnitudes)

#All functions used throughout
def reduced_chi_lower(df):#The lower limit of reduced chi^2 within 1-sigma
    lower_limit=1-1*np.sqrt(2/(df))
    return lower_limit

def reduced_chi_upper(df):#The upper limit of reduced chi^2 within 1-sigma
    upper_limit=1+1*np.sqrt(2/(df))
    return upper_limit

def min_chi(j,a,constants,best_chi2):#The brute-gridding method to find minimum chi^2 for step 2
    list_chi22=[]
    for b in constants:
        d_test = b +0.0*log_dpc[a]
        chi2[j] = chisq(log_dpc[a], list_err_d[a], d_test)
        list_chi22.append(chi2[j]) #The reason i create this is to find the error in slope later on it needs to go through every
        if (chi2[j] < best_chi2):#value of chi^2 which will be in that list
            best_chi2 = chi2[j]
            best_slope = b
        j=j+1
    return best_slope,best_chi2,list_chi22

def plot_galaxy(k): #This is how we plot all the log_d for the specified galaxy
    y=0.0*log_dpc[k] + list_slopes[k]
    y2=0.0*log_dpc[k] + list_slopes_no_int[k]
    plt.rcParams['font.family']='cursive'
    x=np.linspace(0,np.size(log_dpc[k]),np.size(log_dpc[k])) #Using arbitrary values for x because x doesnt matter its just to visually show
    plt.plot(x,y2,color='blue',ls='--')#with intrinsic dispersion
    plt.plot(x,y,color='red')#This is plotting the constant without intrinsic dispersion
    plt.text(0, min(log_dpc[k])-0.03, s='RED line=With intrinsic dispersion',fontsize=10,color='red')
    plt.text(0, min(log_dpc[k])-0.05, s='BLUE dashed line=Without intrinsic dispersion',fontsize=10,color='blue')
    plt.scatter(x,log_dpc[k])
    plt.errorbar(x,log_dpc[k],color='blue',yerr=list_err_d[k],ls='none',elinewidth=0.5)
    plt.xlabel('Arbitrary')
    plt.ylabel('log_dpc')
    plt.title('Galaxy {} ({})'.format(k+1,galaxy[k]))
#Removing outliars
def remove_outliars(galaxy,position): #This can remove a logd value from any given galaxy given you know the position
    log_dpc[galaxy]=log_dpc[galaxy].tolist()
    list_logP[galaxy]=list_logP[galaxy].tolist()  #Convert into list so i can use the del function
    del log_dpc[galaxy][position]
    del list_logP[galaxy][position] 
    log_dpc[galaxy]=np.array(log_dpc[galaxy])#Turning it back into an numpy array so i can manipulate it
    list_logP[galaxy]=np.array(list_logP[galaxy])
    return log_dpc[galaxy],list_logP[galaxy]

#Numpy arrays used throughout step 2
df=np.arange(8.0)
gradients=np.arange(6.4,8.4,0.001) #Change the step in the constants to 0.0001 for computer/laptop to take off
chi2=np.arange(10000000.0)
int_error=np.arange(8.0)*0
best_chi2=1.e5
list_chi=np.arange(8.0)
list_slopes=np.arange(8.0)
list_upper=np.arange(8.0)
list_lower=np.arange(8.0)
list_galaxies_outside=np.arange(8)*0
list_slopes_no_int=np.arange(8.0)

#Calculating/Processing data 
M=alpha*(list_logP+x_shift)+beta #You add the x shift from step 1 to minimise corellation between alpha and beta
log_dpc=(list_m-M+5-err_A)/5

#Removing an outliar in galaxy 4
remove_outliars(3,6)

for i in range(8):
    #Degrees of freedom for each galaxy
    df[i]=len(log_dpc[i])-1
    #Error propagation(without intrinsic dispersion)
    list_err_d[i]=0.2*np.sqrt(((list_logP[i]+x_shift)**2)*(err_alpha**2)+(err_beta**2))
    
    best_slopes,best_chis,list_chi2[i]=min_chi(0,i,gradients,best_chi2)#Using this function to find minimum chi^2
    list_chi[i]=best_chis
    list_slopes[i]=best_slopes
    list_slopes_no_int[i]=best_slopes#Used to plot the dashed line later on
    list_lower[i]=reduced_chi_lower(df[i])
    list_upper[i]=reduced_chi_upper(df[i])
    print('For galaxy',i+1,'Lower limit','       Actual value','      Upper Limit')
    print('          ',reduced_chi_lower(df[i]),'|',best_chis/df[i],'|',reduced_chi_upper(df[i]))
    print('=========================================================================')
    if list_chi[i]/df[i]>list_upper[i] or list_chi[i]/df[i]<list_lower[i]:#If the value of red chi is outside the range then it will put it on a list
        list_galaxies_outside[i]=i
        print('This galaxy falls outside the range: Galaxy',i+1,'with red chi^2 value of',best_chis/df[i] )
print('=========================================================================')

for i in list_galaxies_outside:#I run this to find the intrinsic dispersion for the galaxies outside the 1 sigma reduced chi
    z=0 #This will automatically keep going until the intrinsic dispersion is large enough to make the red chi fall inside the 1-sigma limit
    statement=False
    while statement==False:
        z=z+1
        int_error[i]=int_error[i]+0.001
        #Error propagation(with intrinsic dispersion)
        list_err_d[i]=np.sqrt((0.2*np.sqrt(((list_logP[i]+x_shift)**2)*(err_alpha**2)+(err_beta**2)))**2+int_error[i]**2)
        p,best_chis,list_chi2[i]=min_chi(0,i,gradients,best_chi2) 
        if best_chis/df[i]<list_upper[i]:#This is still within 1 root(2/N-M) of reduced chi for a good model
            statement=True
            list_chi[i]=best_chis
            list_slopes[i]=p
    print('Shifted Galaxy {}({}) with new red chi {} after {} attempts and new distance {}'.format(i+1,galaxy[i],best_chis/df[i],z,10**list_slopes[i]))
#All numpy arrays for finding the error in distance    
error_logd=np.arange(8.0)
error_D=np.arange(8.0)
logd_low=np.arange(8.0)*0
logd_high=np.arange(8.0)*0
logd_err1=np.arange(8.0)*0
logd_err2=np.arange(8.0)*0
logd_err=np.arange(8.0)*0

for i in range(8):
    logd_low[i] = list_slopes[i] + 100.0
    logd_high[i] =list_slopes[i]- 100.0
    j=0
    plt.clf
    plot_galaxy(i)#Plotting all the galaxies
    plt.show()
    for u in gradients:    #Finding the error in constant
        if (list_chi2[i][j] <= (list_chi[i] + 1.0)):
            if (u < logd_low[i]):
                logd_low[i] = u
            if (u > logd_high[i]):
                logd_high[i] = u
        j=j+1   

    logd_err1[i] = logd_high[i] - list_slopes[i]
    logd_err2[i] = list_slopes[i] - logd_low[i]
    if logd_err1[i]>=logd_err2[i]:#Picking the higher error out of the 2 cuz why not
        logd_err[i]=logd_err1[i]
    elif logd_err1[i]<=logd_err2[i]:
        logd_err[i]=logd_err2[i]
    dpc=10**(list_slopes[i])
    error_D[i]=logd_err[i]*(np.log(10)*(10**(list_slopes[i])))
    print('Value for distance(parsec)',dpc,'for Galaxy',galaxy[i],'({})'.format(i+1),'with error',error_D[i])
    
#Finding distance modulii and their errors

distance_modulus=5*list_slopes-5+err_A
tot_err_dmod=(5*logd_err)
x=np.arange(0,24.0,3.0)#Made the steps bigger so the names of the galaxies would be more visible
plt.clf()
for i in range(8):
    plt.annotate('{}'.format(galaxy[i]),xy=(x[i],distance_modulus[i]),fontsize=9)
    plt.annotate('{}'.format(i+1),xy=(x[i],distance_modulus[i]-0.2),fontsize=9)#Plotting each correspoding galaxy number
plt.scatter(x,distance_modulus)#Plotting the distane modulii against arbitrary numbers to visually see them
plt.title('Distances of galaxies')
plt.xlabel('Arbitrary units')
plt.ylabel('Distance Modulus')
plt.errorbar(x,distance_modulus, yerr=tot_err_dmod,ls='none', color='red')
plt.show()

print()
print('STEP 3 AND 4:')
print()

# =============================================================================
# This is step 3 and 4
# =============================================================================

list_d= 10**list_slopes
list_d=list_d*10**-6 #This is Mpc
error_D=error_D*10**-6
df=7#Degrees of freedom

def chisqxy(y, sig_y,sig_x, y_exp,m): #Function to find the Chi^squared value but using both x and y
    chi_2 = np.sum(((y-y_exp)**2.0)/((sig_y**2.0)+((m**2.0)*sig_x**2.0)))
    return chi_2


ndat=1000000
gradients=np.arange(45.0,75,0.001)
err_rvel=157 #This is my starting point on the intrinsic dispersion
best_chi2=1.e5
chi2 =np.arange(ndat)

statement=False
while statement==False:#Automatically goes until found an error in rvel(intrin disp) 
    err_rvel=err_rvel+1 # such that the red chi value is within 0.1 of 1
    j=0
    for m in gradients:
        r_test = m*list_d
        chi2[j] = chisqxy(rvel, err_rvel,error_D, r_test,m)
        if (chi2[j] < best_chi2):    #value of chi^2 which will be in that list
            best_chi2 = chi2[j]
            if np.absolute(best_chi2/df-1)<=0.1:
                statement=True #Statement becomes true to break the loop when right error in rvel is found
                best_slope1=m
                best_chi1=best_chi2
        j=j+1
h_low=best_slope1 + 100
h_high=best_slope1 -100
j=0
for m in gradients:    
    if (chi2[j] <= (best_chi1 + 1.0)):#Finding the error in gradient  
        if (m < h_low):
            h_low = m
        if (m > h_high):
            h_high = m
    j=j+1 

err_h1=h_high-best_slope1
err_h2=best_slope1-h_low

if err_h1>err_h2:
    tot_err=err_h1
elif err_h2>err_h1:#Choosing the bigger error out of the 2 errors
    tot_err=err_h2
print('=============================')
print('Value of H0:',np.round(best_slope1,5))
print('With error on H0: +/-',np.round(tot_err,5))
print('With reduced chi^2:',best_chi1/df)
print('With intrinsic error on rvel:',err_rvel)
print('=============================')

#The whole of step 4:
time=31536000
h0=1/(best_slope1/(3.086e19))
error_1_h0=(1/3.086e19)*tot_err
error_1_h0=error_1_h0*(h0**2)
print('Age of the universe:')
print(np.round((h0/(time*1.e9)),5),'Billion Years')
print('With error: +/-',np.round((error_1_h0/(time*1.e9)),5), 'Billion Years')
print('=============================')
#All the plotting
plt.clf()
plt.rcParams['font.family']='cursive'
x=np.linspace(0,25)
plt.plot(x,LP_Relation(best_slope1,x,0),color='red')
plt.plot(x,LP_Relation(h_low,x,0),ls='--',color='blue',lw=1)
plt.plot(x,LP_Relation(h_high,x,0),ls='--',color='blue',lw=1)
plt.errorbar(list_d,rvel, xerr=error_D, yerr=err_rvel, ls='none')

for i in range(8): #Plotting galaxy number
    c=list_d[i]
    d=rvel[i]
    plt.annotate('{}'.format(i+1),xy=(c,d),fontsize=15)
plt.scatter(list_d,rvel,marker='x',color='black')
plt.xlabel('Distance From Galaxy(Mpc)')
plt.ylabel('Recession Velocity (km/s)')
plt.title('Expansion Rate Of The Universe')
plt.show()



