from scipy.optimize import curve_fit
import scipy.stats as stats
import cv2
import numpy as np
import trackpy as tp
import os
import pims
import tifffile
import matplotlib.pyplot as plt
import pickle
import warnings
import glob

def variance_of_laplacian(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return laplacian.var()

focus_threshold = 100
pixel_to_micron = 0.111

filesA = glob.glob('DepA/*.tif')
filesA.sort()
filesA = filesA[2:21]
framesA = pims.as_grey(pims.ImageSequence(filesA))

frame_area = framesA[0].shape[0] * framesA[0].shape[1]  # height × width in pixels
frame_area_micron = (frame_area * pixel_to_micron**2)

filtered_framesA = [frame for frame in framesA if variance_of_laplacian(frame) > focus_threshold]
featuresA = tp.batch(filtered_framesA, diameter=9, minmass=100, separation=14)
density_per_frameA = featuresA.groupby("frame").size() / frame_area_micron

filesB = glob.glob('DepB/*.tif')
filesB.sort()
filesB = filesB[1:len(filesB)]
framesB = pims.as_grey(pims.ImageSequence(filesB))
filtered_framesB = [frame for frame in framesB if variance_of_laplacian(frame) > focus_threshold]
featuresB = tp.batch(filtered_framesB, diameter=9, minmass=100, separation=14)
density_per_frameB = featuresB.groupby("frame").size() / frame_area_micron
density_per_frameB.plot(marker="o", linestyle="-")
plt.xlabel("Frame")
plt.ylabel("Planar Density (particles/pixel²)")
plt.title("Planar Density Over Time")
plt.savefig("densB.png")
plt.close()

filesC = glob.glob('DepC/*.tif')
filesC.sort()
filesC = filesC[1:len(filesC)]
framesC = pims.as_grey(pims.ImageSequence(filesC))
filtered_framesC = [frame for frame in framesC if variance_of_laplacian(frame) > focus_threshold]
featuresC = tp.batch(filtered_framesC, diameter=9, minmass=100, separation=14)
density_per_frameC = featuresC.groupby("frame").size() / frame_area_micron
density_per_frameC.plot(marker="o", linestyle="-")
plt.xlabel("Frame")
plt.ylabel("Planar Density (particles/pixel²)")
plt.title("Planar Density Over Time")
plt.savefig("densC.png")
plt.close()

print("Frame Density A")
print(density_per_frameA)
print("Frame Density B")
print(density_per_frameB)
print("Frame Density C")
print(density_per_frameC)


mEff = 2.89e-17 
g = 9.8
T = 296
micStepA = 1e-5 
micStepB = 5e-6
k = 1.380e-23

def decay(h,A,s):
    return A*np.exp(-s*h)

hA = np.array([i*micStepA for i in range(len(density_per_frameA))])
lndA = np.array([np.log(density_per_frameA[i]) for i in range(len(density_per_frameA))])
plt.xlabel("Height")
plt.ylabel("ln(planDen)")
plt.scatter(hA,lndA)
plt.savefig("linA.png")
plt.close()
dA = np.array([density_per_frameA[i] for i in range(len(density_per_frameA))])
paramsA = curve_fit(decay,hA,dA)
print(f"k = {mEff*g/(T*paramsA[0][1])} with err {mEff*g/(T*paramsA[0][1]**2)*np.sqrt(np.diag(paramsA[1])[1])}")
heights = np.linspace(0, micStepA * 50,1000)
plt.xlabel("height")
plt.ylabel("Planar Density (particles/pixel²)")
plt.title("Planar Density per Height")
plt.scatter(hA,dA)
plt.plot(heights,decay(heights,paramsA[0][0],paramsA[0][1]))
plt.savefig("densA.png")
plt.close()

'''
slopeA, interceptA, r_valueA, p_valueA, std_errA = stats.linregress(hA, lndA)
print(f"DepA Data")
print(f"k = {-1*mEff*g/(T*slopeA)}")
print(f"Slope: {slopeA}")
print(f"Intercept: {interceptA}")
print(f"R-squared: {r_valueA**2}")
print(f"P-value: {p_valueA}")
print(f"Standard error: {std_errA}")
'''


hB = np.array([i*micStepB for i in range(len(density_per_frameB))])
lndB = np.array([np.log(density_per_frameB.iloc[i]) for i in range(len(density_per_frameB))])
plt.xlabel("Height")
plt.ylabel("ln(planDen)")
plt.scatter(hB,lndB)
plt.savefig("linB.png")
slopeB, interceptB, r_valueB, p_valueB, std_errB = stats.linregress(hB, lndB)
dB = np.array([density_per_frameB.iloc[i] for i in range(len(density_per_frameB))])
paramsB = curve_fit(decay,hB,dB)
print(f"k = {mEff*g/(T*paramsB[0][1])} with err {mEff*g/(T*paramsB[0][1]**2)*np.sqrt(np.diag(paramsB[1])[1])}")

'''
print(f"DepB Data")
print(f"k = {-1*mEff*g/(T*slopeB)}")
print(f"Slope: {slopeB}")
print(f"Intercept: {interceptB}")
print(f"R-squared: {r_valueB**2}")
print(f"P-value: {p_valueB}")
print(f"Standard error: {std_errB}")
'''

hC = np.array([i*micStepB for i in density_per_frameC.index])
lndC = np.array([np.log(density_per_frameC.iloc[i]) for i in range(len(density_per_frameC))])
plt.xlabel("Height")
plt.ylabel("ln(planDen)")
plt.scatter(hC,lndC)
plt.savefig("linC.png")
slopeC, interceptC, r_valueC, p_valueC, std_errC = stats.linregress(hC, lndC)
dC = np.array([density_per_frameC.iloc[i] for i in range(len(density_per_frameC))])
paramsC = curve_fit(decay,hC,dC)
print(f"k = {mEff*g/(T*paramsC[0][1])} with err {mEff*g/(T*paramsC[0][1]**2)*np.sqrt(np.diag(paramsC[1])[1])}")


'''
print(f"DepC Data")
print(f"k = {-1*mEff*g/(T*slopeC)}")
print(f"Slope: {slopeC}")
print(f"Intercept: {interceptC}")
print(f"R-squared: {r_valueC**2}")
print(f"P-value: {p_valueC}")
print(f"Standard error: {std_errC}")
'''
print(f"sB = {paramsB[0][1]} with err {np.sqrt(np.diag(paramsB[1])[1])}")

print(f"Expected Value of slope: {-1*mEff*g/(T*k)}")
