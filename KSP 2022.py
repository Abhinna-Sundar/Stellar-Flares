#!/usr/bin/env python
# coding: utf-8

# In[292]:


import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.io import fits
from glob import glob
from astropy.time import Time
from astropy.timeseries import TimeSeries, aggregate_downsample
from astropy.convolution import convolve, Box1DKernel
import math
from scipy.optimize import curve_fit as cf
from scipy.special import erf
import scipy.special
import sympy as smp
from scipy.integrate import quad
from sympy import *
from numpy.random import lognormal

plt.style.reload_library()
plt.style.use(['science','notebook','grid'])


# ## Function Definition

# In[294]:


get_ipython().run_line_magic('matplotlib', 'qt')


# In[298]:


'''Accessing all the .lc files , binning and returning as Count Rate vs Time of the Day (s)'''

def access_files(filenames, bintime):  
    hdul= fits.open(filename)
    data=hdul[1].data
    mjd         = data['time']*u.s
    cts_per_sec = data['rate']
    cts_error   = data['error']
    if len(cts_per_sec)<=1:
        pass
    else:
        dat = TimeSeries(time=Time(mjd, format="mjd"))
        dat.add_column(cts_per_sec, name="cts_per_sec")
        dat.add_column(cts_error, name="cts_error")
        dat_binned = aggregate_downsample(dat, time_bin_size=bintime*u.s)            
        dat_binned_sec = (dat_binned.time_bin_start.value- dat_binned.time_bin_start.value[0])*86400
    
    return dat_binned, dat_binned_sec


'''Smoothing the datavalues'''

def smooth(data, kernelsize):
    data = convolve(data, Box1DKernel(kernelsize))
    return data

'''Adjacent Averaging method for smoothening'''

def adj_aver(arr):
    for i in range(len(arr)):
        if i ==0:
            arr[i] = (arr[i+1] + arr[i])/2
        elif i == len(arr)-1:
            arr[i] = (arr[i-1] + arr[i])/2
        else:
            arr[i] = (arr[i-1] + arr[i] + arr[i+1])/3
    return arr



'''Checking the end time condition'''

def check(lis, thresh):
    for i in lis:
        if i< thresh:
            return True
    return False 


'''End times extraction based on different conditions:
Four steps must be implemented to find the end times of the flares. 
1. First, the flare end time should be less than that of the next flare start time; 
use this as the limiting condition, but in this case, it should not be the last flare as no next flare exists. 
2. Second, after we are in between the 2 flares, the times should also occur after the flare peaks. 
3. Now, you find the same level count rate as that of the start. 
4. If that does not exist, take the end toime to be 
the start time of the next flare.
'''    
    
def end_times(dat_binned_sec, dat_binned, flares_start_times, flares_peaks_times):
    decay=[]
    decay_times=[]
    end_decay=[]
    end_decay_times=[]
    
    flares_end_times=[]
    
    dat_binned_counts = dat_binned['cts_per_sec'].tolist()
    dat_binned_sec = dat_binned_sec.tolist()
    
    for j in range(0, len(flares_start_times)):
        if j < len(flares_start_times)-1:
            decay = dat_binned_counts[dat_binned_sec.index(flares_peaks_times[j]): dat_binned_sec.index(flares_start_times[j+1])]
            decay_times = dat_binned_sec[dat_binned_sec.index(flares_peaks_times[j]): dat_binned_sec.index(flares_start_times[j+1])]
            
            if check(decay,dat_binned_counts[dat_binned_sec.index(flares_start_times[j])]) == True:      
                for k in range(0, len(decay)):
                    if decay[k]< dat_binned_counts[dat_binned_sec.index(flares_start_times[j])]:
                        flares_end_times.append(decay_times[k-1])
                        break
            else:
                flares_end_times.append(flares_start_times[j+1])
        
        else:
            end_decay = dat_binned['cts_per_sec'][dat_binned_sec.index(flares_peaks_times[j]): len(dat_binned_sec)-1]
            end_decay_times= dat_binned_sec[dat_binned_sec.index(flares_peaks_times[j]): len(dat_binned_sec)-1]
            
            if check(end_decay,dat_binned_counts[dat_binned_sec.index(flares_start_times[j])]) == True:
                for k in range(0, len(end_decay)):
                    if end_decay[k]< dat_binned_counts[dat_binned_sec.index(flares_start_times[j])]:
                        flares_end_times.append(end_decay_times[k-1])
                        break
        
            else:
                flares_end_times.append(end_decay_times[-1])

    return flares_end_times                    
                                                                                    


'''Detection of Flares, start and end times'''

def peaks_and_start_times(dat_binned_sec, dat_binned, slope):
    
    flares_peaks_times=[]
    flares_peaks_counts=[]
    flares_start_times=[]
    flares_index=[]

    for i in range(len(dat_binned_sec)-3):
            if (dat_binned['cts_per_sec'][i]< dat_binned['cts_per_sec'][i+1]< dat_binned['cts_per_sec'][i+2]<dat_binned['cts_per_sec'][i+3]) and (dat_binned['cts_per_sec'][i+3]>=slope*dat_binned['cts_per_sec'][i]):
                for j in range(i+3,len(dat_binned_sec)-3): 
                    if (dat_binned['cts_per_sec'][j]>dat_binned['cts_per_sec'][j+1]>dat_binned['cts_per_sec'][j+2]>dat_binned['cts_per_sec'][j+3]):
                        m = np.argmax(dat_binned['cts_per_sec'][i+3:j+1])
                        
                        ''' finding flare peaks, start and end times for the flares '''
                        
                        if dat_binned_sec[i+3+m] not in flares_peaks_times:
                            flares_peaks_times.append((dat_binned_sec[i+3+m]))
                            flares_peaks_counts.append(dat_binned['cts_per_sec'][i+3+m])
                            flares_start_times.append(dat_binned_sec[i])
                            #flares_index.append(list(np.arange(i, i+3+m)))
                        break 
    flares_end_times = end_times(dat_binned_sec, dat_binned, flares_start_times, flares_peaks_times)
    for i in range(len(flares_start_times)):
        dat_binned_times_list = dat_binned_sec.tolist()
        flares_index.append(list(np.arange(dat_binned_times_list.index(flares_start_times[i]), dat_binned_times_list.index(flares_end_times[i])+1, 1)))
    #flares_index=list(flatten(flares_index))
                      
    return flares_peaks_times, flares_peaks_counts, flares_start_times, flares_end_times, flares_index



'''Plotting the binned and smoothened datavalues, flare peaks (cleaned), start times, end times and background level'''

def plotdata(filename,x,y,flares_peaks_times, flares_peaks_counts, starts, ends, bg_limit):
    plt.figure(figsize=[15,6])
    plt.title(filename[-21:-13])
    plt.scatter(x,y,color='C0', s=7, label='data')
    #plt.plot(x,y,color='r')
    plt.scatter(flares_peaks_times,flares_peaks_counts, color='red', label='Flares peak')
    #[plt.axvline(_x, linewidth=2, linestyle='--',color='magenta') for _x in starts]
    #[plt.axvline(_x, linewidth=2, linestyle='--',color='black') for _x in ends]
    #plt.axhline(bg_limit, linewidth=3, color='limegreen', label='Background Level')
    plt.ylabel('Count Rate', fontsize=15)
    plt.xlabel('Time of the Day (s)',fontsize=15)
    plt.legend()
    plt.show()
    
'''Individaul Flares plotting''' 
    
def indi_flares(dat_binned_sec, dat_binned, flares_start_times, flares_end_times):
    dat_binned['cts_per_sec'] = dat_binned['cts_per_sec'].tolist()
    dat_binned_sec = dat_binned_sec.tolist()
    for i in range(0, len(flares_start_times)):
        plt.figure(figsize=[10,4])
        x = dat_binned_sec[dat_binned_sec.index(flares_start_times[i]): dat_binned_sec.index(flares_end_times[i])+1]
        y = dat_binned['cts_per_sec'][dat_binned_sec.index(flares_start_times[i]): dat_binned_sec.index(flares_end_times[i])+1]
        plt.scatter(x,y,s=5)
        
    
'''Background Extraction
1. Background =  total -  flares (1st time) -------constant background
2. Remove points whose peak points are above 3 sigma level above the background and get the mean background count rate.
3. Repeat step 2 two-three times again until the bg level converges.
4. Check which flares are remaining and return them.
5. Get the final background level.

Update the start and end times of the remaining flares'''

def remove_flares(dat_binned_sec, dat_binned, flares_index):
    index = list(set(flatten(flares_index)))
    bg_time = np.delete(dat_binned_sec, index)
    bg_cts = np.delete(dat_binned['cts_per_sec'], index)
    bg_limit = np.mean(bg_cts)+ 3*np.std(bg_cts)
    
    return bg_time, bg_cts, bg_limit


'''Choosing flares above backgound level'''

def choose_flares_above_bg(flares_peaks_times,flares_peaks_counts,flares_start_times,flares_end_times,flares_index,bg_limit):
    ind=[]
    for i in range(len(flares_peaks_times)):
        if flares_peaks_counts[i]<bg_limit:
            ind.append(i)
        else:
            pass
        flares_peaks_times_n = list(np.delete(flares_peaks_times, ind))
        flares_peaks_counts_n = list(np.delete(flares_peaks_counts, ind))
        flares_start_times_n = list(np.delete(flares_start_times, ind))
        flares_end_times_n = list(np.delete(flares_end_times, ind))
        flares_index_n = list(np.delete(flares_index, ind))
                    
    return flares_peaks_times_n,flares_peaks_counts_n,flares_start_times_n,flares_end_times_n,flares_index_n   


'''Iterate # of times for cleaning background level'''

def bg_extract(how_many_times, flares_peaks_times,flares_peaks_counts,flares_start_times,flares_end_times,flares_index):
    for i in range(how_many_times):
        bg_time, bg_cts, bg_limit = remove_flares(dat_binned_sec, dat_binned, flares_index)
        print('bg level {} = '.format(i), bg_limit, ', bg mean {} = '.format(i), np.mean(bg_cts), ', bg std {} = '.format(i), np.std(bg_cts))
        flares_peaks_times,flares_peaks_counts,flares_start_times,flares_end_times,flares_index = choose_flares_above_bg(flares_peaks_times,flares_peaks_counts,flares_start_times,flares_end_times,flares_index,bg_limit)
        
    return bg_time, bg_cts, bg_limit, flares_peaks_times,flares_peaks_counts,flares_start_times,flares_end_times,flares_index    


'''Update the start and end times above background level'''

def update_starts_ends(dat_binned_sec, dat_binned, flares_peaks_times, flares_peaks_counts, flares_start_times, flares_index, bg_limit, param_limit):
    flares_cleaned_times=[]
    flares_cleaned_counts=[]
    flares_starts=[]
    flares_ends=[]
    flares_cleaned_peaks_times=[]
    flares_cleaned_peaks_counts=[]
    
    for p in range(len(flares_start_times)):
        counts = [dat_binned['cts_per_sec'][j] for j in flares_index[p]]
        times = [dat_binned_sec[j] for j in flares_index[p]]
        inds = [i for i,v in enumerate(counts) if v > bg_limit]
        flares_counts = [counts[k] for k in inds]
        flares_times = [times[k] for k in inds]
        if len(flares_times)> param_limit:     # 10 parameter limit
            flares_cleaned_counts.append(flares_counts)
            flares_cleaned_times.append(flares_times) 
            flares_cleaned_peaks_times.append(flares_peaks_times[p])
            flares_cleaned_peaks_counts.append(flares_peaks_counts[p])
        else:
            pass
    
    for i in range(len(flares_cleaned_times)):
        flares_starts.append(flares_cleaned_times[i][0])
        flares_ends.append(flares_cleaned_times[i][-1])
        
    return flares_starts, flares_ends, flares_cleaned_peaks_times, flares_cleaned_peaks_counts, flares_cleaned_times, flares_cleaned_counts    


'''Defining Convolution function for flares fitting'''

def convolve_exp_norm(x, A, mu, sigma, alpha, bg):   ## alpha = lambda here
    co = A * np.exp( alpha*mu+ (alpha**2) *(sigma**2)/2.0)
    x_erf = (x - mu - alpha*sigma**2)/(np.sqrt(2.0)*sigma)
    y = co * np.exp(-alpha*x) * (1.0 + scipy.special.erf(x_erf)) + bg
    return y

'''Lognormal function for flares'''

def LogNormal(x, A, mu, sigma, bg):
    return A/(x*sigma*np.sqrt(2*np.pi)) * np.exp(-(((np.log(x)-mu)**2)/(2*sigma**2))) + bg


'''FRED fits'''

def FRED(x, A, tau1, tau2, bg):
    return A*(np.exp(2*np.sqrt(tau1/tau2)))*np.exp(-tau1/x-x/tau2) + bg


'''Plotting the data with fitted convolution funtions on the detected flares'''

def plot_final(filename, dat_binned, dat_binned_sec, flares_cleaned_peaks_times, flares_cleaned_peaks_counts, flares_cleaned_times, flares_cleaned_counts, flares_starts, flares_ends, bg_limit):
    plt.figure(figsize=[15,6])
    #plt.errorbar(np.array(dat_binned_sec), np.array(dat_binned['cts_per_sec']), yerr=np.array(dat_binned['cts_error']), fmt='.', color='C0', label='data') 
    plt.scatter(np.array(dat_binned_sec), np.array(dat_binned['cts_per_sec']), s=10, color='C0', label='data')
    #plt.scatter(flares_cleaned_times,flares_cleaned_counts, s=20, color='g')
    plt.scatter(flares_cleaned_peaks_times, flares_cleaned_peaks_counts, color='red', label='flare peaks')
    for i in range(len(flares_cleaned_peaks_times)):
        #p_opt, p_cov = cf(convolve_exp_norm, np.array(flares_cleaned_times[i]) , np.array(flares_cleaned_counts[i]),(flares_cleaned_peaks_counts[i],flares_cleaned_peaks_times[i], 1000, 0.001, bg_limit))
        p_opt, p_cov = cf(LogNormal, np.array(flares_cleaned_times[i]) , np.array(flares_cleaned_counts[i]),(flares_cleaned_peaks_counts[i],flares_cleaned_peaks_times[i], 0.1, bg_limit))
        #p_opt, p_cov = cf(FRED, np.array(flares_cleaned_times[i]) , np.array(flares_cleaned_counts[i]),(400, 3900500,120, bg_limit))
        #plt.plot(np.array(flares_cleaned_times[i]), convolve_exp_norm(np.array(flares_cleaned_times[i]), *p_opt), color='red',linewidth=4, alpha=0.7,  label='Fit_GaussExp_convolve')
        plt.plot(np.array(flares_cleaned_times[i]), LogNormal(np.array(flares_cleaned_times[i]), *p_opt), color='red',linewidth=4, alpha=0.7,  label='Fit_LogNormal')        
        #plt.plot(np.array(flares_cleaned_times[i]), FRED(np.array(flares_cleaned_times[i]), *p_opt), color='red',linewidth=4, alpha=0.7,  label='Fit_FRED')        
        plt.axvline(flares_starts[i], linewidth=2, linestyle='--',color='magenta')
        plt.axvline(flares_ends[i], linewidth=2, linestyle='--',color='black')
        plt.show()
    plt.axhline(bg_limit, linewidth=3, color='limegreen', label='Background + 3 $\sigma$')
    plt.ylabel('Count rate')
    plt.xlabel('Time of the day (s)')
    plt.title(filename[31:39])
    plt.legend()
    


# In[299]:


path='E:/Krittika Internship/'
filenames = sorted(glob(path+'*.lc'))

for filename in filenames:  
    bintime = 125 # binning time
    kernelsize=8 # Kernel size for smoothening
    slope=1.08 # what slope you require i.e. 4th point/1st point (Count rate)
    how_many_times=3 # how many times you want to iterate for background estimation
    param_limit= 15 # how many datapoints you set as limit for a flare (this is imp for fitting)
    
    dat_binned, dat_binned_sec = access_files(filenames, bintime)  # Accessing files and binning them (bintime in seconds)
    print('filename = ',filename[31:39])
    dat_binned['cts_per_sec'] = smooth(dat_binned['cts_per_sec'], kernelsize)  # Smoothening the datavalues using Boxcar
    dat_binned['cts_error'] = smooth(dat_binned['cts_error'], kernelsize)
    flares_peaks_times, flares_peaks_counts, flares_start_times, flares_end_times, flares_index = peaks_and_start_times(dat_binned_sec, dat_binned, slope)    
    #plotdata(filename,dat_binned_sec,dat_binned['cts_per_sec'],flares_peaks_times, flares_peaks_counts, flares_start_times, flares_end_times, bg_limit)
    bg_time, bg_cts, bg_limit, flares_peaks_times,flares_peaks_counts,flares_start_times,flares_end_times,flares_index = bg_extract(how_many_times, flares_peaks_times,flares_peaks_counts,flares_start_times,flares_end_times,flares_index)
    flares_starts, flares_ends, flares_cleaned_peaks_times, flares_cleaned_peaks_counts, flares_cleaned_times, flares_cleaned_counts = update_starts_ends(dat_binned_sec, dat_binned, flares_peaks_times, flares_peaks_counts, flares_start_times, flares_index, bg_limit, param_limit)
    plot_final(filename, dat_binned, dat_binned_sec, flares_cleaned_peaks_times, flares_cleaned_peaks_counts, flares_cleaned_times, flares_cleaned_counts, flares_starts, flares_ends, bg_limit)
    #plt.savefig(str(filename)+'.pdf', dpi=600)


# In[ ]:





# In[27]:





# In[49]:


def SNR(time, counts, starttime, endtime, bintime):
    S_B = counts[time.index(starttime):time.index(endtime)+1]  #Signal+bkg for flare region
    B = counts[time.index(starttime)]  #assumed Background to be flare starttime countrate
    S = [S_B - B for x in S_B] ## only signal
    
    return np.mean(S)*np.sqrt(bintime)/(np.sqrt(np.mean(S)+2*B))    #SNR = S/sqrt(S+2B)


# In[51]:


SNR(list(dat_binned_sec), list(dat_binned['cts_per_sec']), flares_cleaned_times[5][0], flares_cleaned_times[5][-1], 125)


# In[287]:


from matplotlib import gridspec

t = flares_cleaned_times[0]#[:100]
a = flares_cleaned_counts[0]#[:100]

#p_opt, p_cov = cf(FRED, np.array(t) , np.array(a),(flares_cleaned_peaks_counts[0], 3900500,120, bg_limit))     
#p_opt, p_cov = cf(LogNormal, np.array(flares_cleaned_times[0]) , np.array(flares_cleaned_counts[0]),(flares_cleaned_peaks_counts[0],10, 0.1, bg_limit))
p_opt, p_cov = cf(convolve_exp_norm, np.array(flares_cleaned_times[0]) , np.array(flares_cleaned_counts[0]),(flares_cleaned_peaks_counts[0],flares_cleaned_peaks_times[0], 1000, 0.001, bg_limit))
chi=[]
for i in range(len(t)):
    if a[i] != 0:
        chi.append((a[i]-convolve_exp_norm(np.array(t), *p_opt)[i])**2/(a[i]))
    chisq = sum(chi)
    contri= (a-convolve_exp_norm(np.array(t), *p_opt))**2/(a)
    
#plt.scatter(t,chi,s=5) 

fig = plt.figure(figsize=[10,8])
spec = gridspec.GridSpec(ncols=1, nrows=2,hspace=0.5, height_ratios=[2, 1])
ax0 = fig.add_subplot(spec[0])
ax0.set_title('20200311')
ax0.scatter(t, a,s=10)
ax0.set_ylabel('Count Rate')
#ax0.plot(np.array(t), FRED(np.array(t), *p_opt), color='red',linewidth=4, alpha=0.7,  label='Fit_FRED')
#ax0.plot(np.array(t), LogNormal(np.array(t), *p_opt), color='red',linewidth=4, alpha=0.7,  label='Fit_LogNormal')
ax0.plot(np.array(t), convolve_exp_norm(np.array(t), *p_opt), color='red',linewidth=4, alpha=0.7,  label='Fit_ExpGauss_convolve')
ax0.legend()
ax1 = fig.add_subplot(spec[1])
ax1.scatter(t, chi,s=10, label='')
ax1.set_xlabel('Time of the Day (s)')
ax1.set_ylabel(r'$(data-model)^2/err^2$')
ax1.legend()
plt.show()


# In[234]:


get_ipython().system('git clone https://github.com/lupitatovar/Llamaradas-Estelares.git')


# In[236]:


cd Llamaradas-Estelares


# In[237]:


from Flare_model import flare_model


# In[256]:


len(t)


# In[289]:


init_vals=[21122, 1800, 421] #[tpeak, FWHM, amp]
popt, pcov = cf(flare_model, t,a, p0=init_vals,maxfev=10000)

plt.scatter(np.array(t),np.array(a)-bg_limit*np.ones(len(t)),color='black', s=10)
#plt.plot(np.array(t), flare_model(np.array(t), *init_vals),color='red', label='Initial Guess', alpha=0.3)
plt.plot(np.array(t), flare_model(np.array(t), *popt),color='r', label='Model Fit', lw=2)
plt.legend()
plt.xlabel('Time')
plt.ylabel('Count Rate')

popt


# $A e^{2\sqrt{\tau_1/\tau_2})} \cdot e^{-\tau_1/x-x/\tau_2}$ + $bg$ where $\sqrt{\tau_1\tau_2} = peak$ 

# In[48]:


np.mean(b)*np.sqrt(bintime)/(np.sqrt(np.mean(b)+2*a[0]))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'qt')


# In[ ]:





# In[ ]:


plt.plot(dat_binned_sec, dat_binned['cts_per_sec'])


# In[ ]:





# ### Gaussian Function
# 
# $g(x) = A$ $exp(-(x-B)^2/C^2)$
# 
# ### Exponential Function
# 
# $h(x) = exp (-Dx)$
# 
# 
# ### Convolution Function
# 
# $f(t)$ = $\frac{1}{2} \sqrt{\pi}$ $A$ $C$ $exp\left[D(B-t) + \frac{C^2 D^2}{4}\right]\left[erf(Z) - erf(Z-\frac{t}{C})\right]$
# 
# #### where $Z = \frac{2B+ C^2 D}{2C}$ 

# In[230]:


f= A*smp.exp(-((x-B)**2)/(C**2))
g= smp.exp(-(D*(x)))
def convolve(f, g, x, lower_limit, upper_limit):
    t = Symbol('t')
    h = g.subs(x, t - x)
    return integrate(f * h, (t, lower_limit, upper_limit))

t= Symbol('t')
convolve(f, g, x, 0, t)

