# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:36:07 2024

@author: php20jo
"""
import os
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, find_peaks
from scipy.interpolate import splev, splrep

# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))

# Change the current working directory

os.chdir('C:/Users/php20jo/Desktop/CONTIN/dpscience-DCONTINPALS-64edb59/pyDCONTINPALS/Data/')

#os.chdir('C:/Users/php20jo/Desktop/CONTIN/dpscience-DCONTINPALS-64edb59/pyDCONTINPALS/Data/F_24')

#os.chdir('C:/Users/php20jo/Desktop/CONTIN/dpscience-DCONTINPALS-64edb59/pyDCONTINPALS/MnM/')
#os.chdir('C:/Users/php20jo/Desktop/CONTIN/dpscience-DCONTINPALS-64edb59/pyDCONTINPALS/PLA/')
# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))

import numpy as np

CB = ['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00']

#%%
#find FV
#convert lifetime to volume
TaoEldrup=np.load('TaoEldrup.npy') #import TaoEldrup bubble model with 10,000 points (calculated from equation)
TaoEldrup_x=TaoEldrup[0,:] 
TaoEldrup_y=TaoEldrup[1,:]
spl = splrep(TaoEldrup_x, TaoEldrup_y)
FV = splev(1.5,spl) # calculate the FV for the measured oPs lifetime
plt.figure()
plt.plot(TaoEldrup_x,TaoEldrup_y)
plt.xlabel('oPs Lifetime [ps]')
plt.ylabel('Free Volume [$\AA^3$]')
#%%

#cb=2
#pos = 5
run = 1
sample = 3
state = 4
states = ['Initial','Wet','Dried','stored','xdry']



print('sample' + str(sample) + '/' + states[state] + '/run' + str(run) + '.datout.txt')

file = ('sample' + str(sample) + '/' + states[state] + '/run' + str(run) + '.datout.txt')


x,y = np.loadtxt(file)

plt.figure(sample)
plt.plot(x,y,label=states[state],color=CB[state])
plt.xlabel('charateristic lifetimes [ps]')
plt.ylabel('intensity pdf [a.u.]')
#plt.title('sample '+str(sample)+' run '+str(run))
plt.legend()

#%%
plt.figure(3)
file = ('sample3/xdry/8M.datout.txt')
x,y = np.loadtxt(file)

plt.plot(x,y,label='xdry 8M',color=CB[state+1])
plt.legend()
#%%
#a=0
plt.figure('PLA')
file = ('4m.datout.txt')
x,y = np.loadtxt(file)

plt.plot(x,y,label=str(file[:2]),color=CB[a])
plt.xlabel('charateristic lifetimes [ps]')
plt.ylabel('intensity pdf [a.u.]')
plt.legend()
a=a+1
#%%plot av

plt.figure('av_lifetime')
file = ('y_av_initial')
y = np.loadtxt(file)

plt.plot(x,y,label='dried',color=CB[2])
plt.legend()
plt.xlabel('charateristic lifetimes [ps]')
plt.ylabel('intensity pdf [a.u.]')
#%%
#%%plot av free vol rad
n = 2
if n==0:
    file = ('y_av_initial')
    
if n==1:
    file = ('y_av_wet')
        
if n==2:
    file = ('y_av_dried')

y = np.loadtxt(file)
#n=2
#
plt.figure('av_diameter')
x_fv = np.zeros(len(x))
x_fvr = np.zeros(len(x))
for i in range(len(x)):
    x_fv[i] = splev(x[i]/1000,spl)
    x_fvr[i] = np.cbrt((3*x_fv[i])/(4*np.pi))*2
    

plt.plot(x_fvr[:32],y[:32],label=states[n],color=CB[n])
plt.xlim(2.5)
plt.xlabel('Free volume D $\AA$')
plt.ylabel('intensity pdf [a.u.]')
plt.legend()

#%% charlies plot

x_CD, y_CD = np.loadtxt('0.csv',delimiter=',',unpack=True)

x_CW, y_CW = np.loadtxt('2.csv',delimiter=',',unpack=True)

xd_CD = np.zeros(len(x_CD))
xd_CW = np.zeros(len(x_CW))
for i in range(len(x_CD)):
    xd_CD[i] = np.cbrt((3*x_CD[i])/(4*np.pi))*2
    xd_CW[i] = np.cbrt((3*x_CW[i])/(4*np.pi))*2


plt.plot(xd_CD,y_CD/15,label='MD dry',color=CB[3])
plt.plot(xd_CW[:120],y_CW[:120]/15,label='MD Wet',color=CB[4])
plt.legend()
plt.ylabel('PSD')
plt.xlabel('Pore diameter $\AA$')


#%%plot MnM data
#a=0
plt.figure('Moses')
file = ('Moses\MH_1MI.datout.txt')
x,y = np.loadtxt(file)

plt.plot(x,y,label=str(file),color=CB[a])
plt.xlabel('charateristic lifetimes [ps]')
plt.ylabel('intensity pdf [a.u.]')
plt.legend()
a=a+1


#%%
#sample 1 run 1  = pos0 
#sample 1 run 2  = pos1
#sample 2 run 1
#sample 2 run 2
#sample 3 run 1
#sample 3 run 2

#y_big_dried = np.zeros((6,100))

#if using y_big use the txt file

#for i in range(len(y)):
 #   y_big_dried[pos,i] = y[i] 
#%%

##np.savetxt('y_big_dried.txt',y_big_dried)
#%%
#plt.plot(x,y_big_dried[pos,:],label=(file))
#plt.legend()

#%%make average
#y_av_dried = np.zeros(100)
#for i in range(len(y)):
 #   y_av_dried[i] = np.mean(y_big_dried[:,i])
#%%
#plt.plot(x,y_av_dried)
##np.savetxt('y_av_dried',y_av_dried)
#%%
#peaks, _ = find_peaks(y, height=0.001)
#print('peaks = ' , peaks)

maxima = argrelextrema(y, np.greater)
#minima = argrelextrema(y, np.less)
#extrema = np.concatenate([maxima, minima], axis=None)
#print('extrema = ' , extrema)
plt.figure(sample+3)
plt.plot(x,y,label=state,color=CB[cb],marker='x')
#plt.plot(x[maxima],y[maxima],'x')

#plt.plot(x[peaks],y[peaks],'x')
plt.xlabel('charateristic lifetimes [ps]')
plt.ylabel('intensity pdf [a.u.]')
plt.title('sample '+str(sample)+' run '+str(run))
plt.legend()
#print(x[peaks])



#%%
x_fv = np.zeros(len(x))
x_fvr = np.zeros(len(x))
for i in range(len(x)):
    x_fv[i] = splev(x[i]/1000,spl)
    x_fvr[i] = np.cbrt((3*x_fv[i])/(4*np.pi))
    

plt.plot(x_fvr[:32],y_av_initial[:32],label=states[0],color=CB[0])
plt.xlabel('Free volume radius $\AA$')
plt.ylabel('intensity pdf [a.u.]')
plt.legend()
#plt.figure()
#plt.plot(x_fv,y)
#plt.plot(TaoEldrup_x,TaoEldrup_y)

#plt.xlabel('lifetime [ns]')
#plt.ylabel('vol $\AA\u00b3$')
#%%
plt.title('average')
#%% convert lifetime to volume

x_ns = x/1000  #convert to ns

print(x)
#%%
plt.plot(x,y)
#%%

def bubble(R):
    A=(R/(R+1.66))
    l=2*(1-A+(1/(2*np.pi))*np.sin((2*np.pi*A)))
    t=1/l
    return t

R = np.linspace(0,5,100)

t = bubble(R)

V = (4/3)*np.pi*R**3
plt.figure()
plt.plot(t,R,color='red')
plt.xlabel('oPs Lifetime [ps]')
plt.ylabel('FV radius [$\AA$]')
#%%
list_of_peaks.append(file[:-11])
list_of_peaks.append(str(x[peaks]))
print(list_of_peaks)

#%%Franc

r = ['0','0p57','0p8','1']
s = ['s','ns']
d = ['','_dmp30']

file = r[3]+'_'+s[1]+d[0]+'.datout.txt'
print(file)
x,y = np.loadtxt(file)

#plt.plot(x,y)

#plt.figure('s')
x_fv = np.zeros(len(x))
x_fvr = np.zeros(len(x))
for i in range(len(x)):
    x_fv[i] = splev(x[i]/1000,spl)
    x_fvr[i] = np.cbrt((3*x_fv[i])/(4*np.pi))*2
    

plt.plot(x_fv[:32],y[:32],label=file[:-11],linestyle='--')
plt.xlim(2.5)
plt.xlabel('Free volume $\AA^3$')
plt.ylabel('intensity pdf [a.u.]')
plt.legend()

#%%

os.chdir('C:/Users/php20jo/Desktop/CONTIN/dpscience-DCONTINPALS-64edb59/pyDCONTINPALS/Data/IRF/1M/')

file = '240123-22-58_T1.datout.txt'
x,y = np.loadtxt(file)

os.chdir('C:/Users/php20jo/Desktop/CONTIN/dpscience-DCONTINPALS-64edb59/pyDCONTINPALS/Data/sample1/initial/')

file1 = 'run1.datout.txt'
x1,y1 = np.loadtxt(file1)

plt.plot(x,y,label='Si')

plt.plot(x1,y1,label='DGEBA-MXDA')

plt.xlabel('charateristic lifetimes [ps]')
plt.ylabel('intensity pdf [a.u.]')
plt.legend()