import statsmodels.api as sm
import pims
import trackpy as tp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# Link trajectories
features = pd.read_pickle('frameDataC2_18.pkl')
trajectories = tp.link_df(features, search_range=10, memory=3)
trajectories = tp.filter_stubs(trajectories)

def diffus(t,D):
    return 4*D*t

msd = tp.imsd(trajectories, mpp=0.1, fps=20 ,max_lagtime = 1003) 
time = msd.index.to_numpy()
print(time)

meanMSD = msd.mean(axis=1)
D = curve_fit(diffus,time,meanMSD)
print(D)
print(f"k = {D[0]*1E-18*np.pi*0.000931*3/(296)} with err{D[1] * 1e-18*np.pi*0.000931*3/(296)}")
t = np.arange(0,60,.1)
plt.figure()
plt.xlabel('Time (s)')
plt.ylabel('MSD (m^-6)^2')
plt.scatter(time,meanMSD)
plt.plot(t,diffus(t,D[0]))
plt.savefig('DiffusionC2_18.png')
plt.close()

step_displacements = tracjectories.groupby('particle')[['x', 'y']].diff().dropna()

# Compute step magnitude
step_displacements['step_mag'] = np.sqrt(step_displacements['x']**2 + step_displacements['y']**2)

# Bin steps by time lag
step_displacements['frame'] = trajectories['frame']
mean_step_displacement = step_displacements.groupby('frame')['step_mag'].mean()

