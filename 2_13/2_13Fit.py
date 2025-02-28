import statsmodels.api as sm
import pims
import trackpy as tp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# Link trajectories
features = pd.read_pickle('frameData.pkl')
trajectories = tp.link_df(features, search_range=10, memory=3)
trajectories = tp.filter_stubs(trajectories)

def diffus(t,D):
    return 4*D*t

msd = tp.imsd(trajectories, mpp=0.1, fps=10 ,max_lagtime = 1003) 
time = msd.index.to_numpy()
print(time)
   # data.append((t,np.mean(np.array([msd.loc[t,i] for i in msd.columns])),np.var(np.array([msd.loc[t,i] for i in msd.columns]))))

meanMSD = msd.mean(axis=1)
D = curve_fit(diffus,time[:400],meanMSD[time[:400]])
print(D)
print(D[0]*1E-18*np.pi*0.000931*3/(296))
t = np.arange(0,60,.1)
plt.figure()
plt.xlabel('Time (s)')
plt.ylabel('MSD (m^-6)^2')
plt.scatter(time,meanMSD)
plt.plot(t,diffus(t,D[0]))
plt.savefig('Diffusion1.png')
plt.close()


'''
msd = tp.emsd(trajectories,mpp=0.1,fps=10,max_lagtime=500)
print(type(msd))
print(msd)
print(msd.index)
times = np.array(msd.index)
times = times[:,np.newaxis]
disp = np.array(msd[msd.index]*1E-12)
a, _, _, _ = np.linalg.lstsq(times,disp)
print(a)
'''
#model = sm.WLS(time, meanMSD, weights=weightsM)
#results = model.fit()
#print(results.summary())
