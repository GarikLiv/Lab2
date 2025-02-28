import statsmodels.api as sm
import pims
import trackpy as tp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import os
import tifffile
import trackpy as tp
import pickle
import warnings


filesA = glob.glob('DepA/AMot/*.tif')
filesB = glob.glob('DepB/BMot/*.tif')
filesC = glob.glob('DepC/CMot/*.tif')
filesA.sort()
filesB.sort()
filesC.sort()
framesA = pims.as_grey(pims.ImageSequence(files))
framesB = pims.as_grey(pims.ImageSequence(files))
framesC = pims.as_grey(pims.ImageSequence(files))
featuresA = tp.batch(frames, diameter=9, minmass=100, separation=14, invert=False,processes=1)
featuresB = tp.batch(frames, diameter=9, minmass=100, separation=14, invert=False,processes=1)
featuresC = tp.batch(frames, diameter=9, minmass=100, separation=14, invert=False,processes=1)
