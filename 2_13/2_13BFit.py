import statsmodels.api as sm
import pims
import trackpy as tp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def diffus(t,D):
    return 4*t*D

featuresB = pd.read_pickle('frameData2.pkl')
trajectoriesB = tp.link_df(featuresB, search_range=10, memory=3)
trajectoriesB = tp.filter_stubs(trajectoriesB)
msdB = tp.imsd(trajectoriesB, mpp=0.1, fps=10 ,max_lagtime = 695) 
timeB = msdB.index.to_numpy()
meanMSDB = msdB.mean(axis=1)
DB = curve_fit(diffus,timeB,meanMSDB)
print(DB)
print('k = ' + str(DB[0]*1E-18*np.pi*0.00089*3/(298)))
print('err = ' + str(DB[1]*1E-18*np.pi*0.00089*3/(298)))
t= np.arange(0,11,0.1)
plt.figure()
plt.xlabel('Time (s)')
plt.ylabel('MSD (m^-6)^2')
plt.scatter(timeB,meanMSDB,color='black',s=10)
plt.scatter(timeB,[np.log(i) for i in meanMSDB],color='red',s=10)
plt.plot(t,diffus(t,DB[0]))
plt.savefig('Diffusion2.png')
plt.close()
