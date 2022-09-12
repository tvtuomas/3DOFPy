#-----------------------------------------------------------------------------
# testFoamScript
# Example script for multilevel 3D Optical Flow estimation
#
# calls [u,v,w] = OpticalFlow3D(im0, im1, para) with suitable parameters
# for an artificially deformed foam example mentioned in our paper. 
# Calculates an evaluation and saves the result as .raw file, which can  
# be read out by src/readRawVectorfield.m. 
#
# Calls for: 
#            OpticalFlow3D
#           
#
# Design and implementation by:
# Tessa Nogatz <tessa.nogatz@itwm.fraunhofer.de>  
# (c) 2018-2021 Fraunhofer ITWM, Kaiserslautern

import time
import numpy as np
from optical_flow import *
from skimage import io


#Testing 
# Functions to test: getPsiSmooth3D, psiDeriv, warpImage3D, centralDiffDir

test_array1 = np.ones((10,10,10))
test_array1[6:8, 5:7:, 4:6] = 2

test_array2 = np.ones((10,10,10))
test_array2[2:3, 3:4:, 4:5] = 2

u = np.ones((10, 10, 10))
v = np.ones((10, 10, 10)) * 1.5
w = np.ones((10, 10, 10)) * 2.6

Ix, Iy, Iz = centralDiffDir(test_array1)
warped = warpImage3D(test_array1, u, v, w)

deriv = psiDeriv(test_array1)

PsiSmoothEast, PsiSmoothSouth, PsiSmoothBack = getPsiSmooth3D(u, v, w)

#Testing ends

start = time.time()

fullFoam = io.imread('img/fullFoam.tif').astype(float)
defFoam  = io.imread('img/defFoam.tif').astype(float)

fullFoam_p = np.pad(fullFoam, ((3,3),(3,3),(3,3)), 'constant')
defFoam_p  = np.pad(defFoam, ((3,3),(3,3),(3,3)), 'constant')

parameters = getDefaultParameters(fullFoam_p.shape)

u, v, w = OpticalFlow3D(defFoam_p, fullFoam_p, parameters)

u = u[3:-4, 3:-4, 3:-4]
v = v[3:-4, 3:-4, 3:-4]
w = w[3:-4, 3:-4, 3:-4]
    
end = time.time()

print("Done in " + str(end - start) + " s")

#paraRes = evaluateImages(u, v, w, defFoam, fullFoam);

#fin = fopen(strcat('img/testFoam_x1disp_',dimStr,'.raw'), 'w+');
#cnt = fwrite(fin, single(u), 'single');
#fclose(fin);

#fin = fopen(strcat('img/testFoam_x2disp_',dimStr,'.raw'), 'w+');
#cnt = fwrite(fin, single(v), 'single');
#fclose(fin);

#fin = fopen(strcat('img/testFoam_x3disp_',dimStr,'.raw'), 'w+');
#cnt = fwrite(fin, single(w), 'single');
#fclose(fin);
