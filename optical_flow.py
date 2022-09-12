#Python implementation by Tuomas Turpeinen (tuomas.turpeinen@vtt.fi)

import numpy as np
import time
from math import sqrt
from successiveOverrelaxation3D import *
from scipy.ndimage import gaussian_filter, zoom, convolve, median_filter
from scipy.interpolate import RegularGridInterpolator
from skimage.transform import resize
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io

def getDefaultParameters(dimensions):
    #-----------------------------------------------------------------------------
    # getParaFlow(p,q,r) 
    # Sets parameters for 3D Optical Flow estimation
    #
    # para=getParaFlow(p,q,r) sets the mandatory parameters for 3DOF
    # calculation. They are:
    #
    # height, width, length:     size of original images
    # lambda:                    weight of smoothness factor
    # mu:                        weight of gradient non-consistency
    # warps:                     number of iterations. Note that there are very
    #                            costly, often a single warp can already 
    #                            achieve sufficient results
    # outer_iter:                max number for fixed point iteration
    # sor_iter:                  max number for sor iteration
    # downsampling:              downsampling factor
    # sigma:                     smoothing factor for Gaussian filter
    # levels:                    number of coarse-to-fine approximations
    # w:                         weight for successive overrelaxation, 
    #                            strictly below 2
    # median_filtering:          1 if median filter should be applied
    # medianx, mediany, medianz: median filter size
    #          
    # For example use see testFoamScript.
    #
    # Design and implementation by:
    # Tessa Nogatz <tessa.nogatz@itwm.fraunhofer.de>  
    # (c) 2018-2021 Fraunhofer ITWM, Kaiserslautern
    #-----------------------------------------------------------------------------
    parameters = {}
    parameters['height'] = dimensions[1]
    parameters['width'] = dimensions[2]
    parameters['length'] = dimensions[0]
    parameters['lambda'] = 20.0  
    parameters['mu'] = 1.0             
    parameters['warps'] = 2
    parameters['outer_iters'] = 15          
    parameters['sor_iters'] = 15        
    parameters['downsampling'] = .9   
    parameters['sigma'] = 1 / sqrt(2*parameters['downsampling'])
    parameters['levels'] = 10
    parameters['w'] = 1.99
    parameters['median_filtering'] = 1  
    parameters['medianx'] = 5
    parameters['mediany'] = 5
    parameters['medianz'] = 5
    return parameters
    
def getPsiSmooth3D(u, v, w):
    #-----------------------------------------------------------------------------
    # getPsiSmooth3D
    # compute diffusivity of vector field
    #
    # [PsiSmoothEast, PsiSmoothSouth, PsiSmoothBack] = getPsiSmooth3D(u, v, w)
    # calculates the diffusivity based on 6 neighboring voxels. 3D extension of
    # Brox 2D diffusivity
    #
    # Design and implementation by:
    # Tessa Nogatz <tessa.nogatz@itwm.fraunhofer.de>  
    # (c) 2018-2021 Fraunhofer ITWM, Kaiserslautern
    #-----------------------------------------------------------------------------

    k1 = np.array([-1, 1]).reshape(1,1,2)
    ux1 = convolve(u, k1, mode='nearest')
    uy1 = convolve(u, k1.reshape(1, 2, 1), mode='nearest')
    uz1 = convolve(u, k1.reshape(2, 1, 1), mode='nearest')

    k2 = np.array([-0.5, 0, 0.5]).reshape(1,1,3)
    ux2 = convolve(u, k2, mode='nearest')
    uy2 = convolve(u, k2.reshape( 1, 3, 1), mode='nearest')
    uz2 = convolve(u, k2.reshape( 3, 1, 1), mode='nearest')

    k3 = np.array([0.5, 0.5]).reshape(1,1,2)
    uxsq = np.power(ux1, 2) + np.power(convolve(uy2, k3, mode='nearest'), 2) + np.power(convolve(uz2, k3, mode='nearest'), 2)
    uysq = np.power(uy1, 2) + np.power(convolve(ux2, k3.reshape(1, 2, 1), mode='nearest'), 2) + np.power(convolve(uz2, k3.reshape(1, 1, 2), mode='nearest'), 2)
    uzsq = np.power(uz1, 2) + np.power(convolve(ux2, k3.reshape(2, 1, 1), mode='nearest'), 2) + np.power(convolve(uy2, k3.reshape(2, 1, 1), mode='nearest'), 2)

    vx1 = convolve(v, k1, mode='nearest')
    vy1 = convolve(v, k1.reshape(1, 2, 1), mode='nearest')
    vz1 = convolve(v, k1.reshape(2, 1, 1), mode='nearest')

    vx2 = convolve(v, k2, mode='nearest')
    vy2 = convolve(v, k2.reshape(1, 3, 1), mode='nearest')
    vz2 = convolve(v, k2.reshape(3, 1, 1), mode='nearest')

    vxsq = np.power(vx1, 2) + np.power(convolve(vy2, k3, mode='nearest'), 2) + np.power(convolve(vz2, k3, mode='nearest'), 2) 
    vysq = np.power(vy1, 2) + np.power(convolve(vx2, k3.reshape(1, 2, 1), mode='nearest'), 2) + np.power(convolve(vz2, k3.reshape(1, 1, 2), mode='nearest'), 2)
    vzsq = np.power(vz1, 2) + np.power(convolve(vx2, k3.reshape(2, 1, 1), mode='nearest'), 2) + np.power(convolve(vy2, k3.reshape(2, 1, 1), mode='nearest'), 2)
             
    wx1 = convolve(w, k1, mode='nearest')
    wy1 = convolve(w, k1.reshape(1, 2, 1), mode='nearest')
    wz1 = convolve(w, k1.reshape(2, 1, 1), mode='nearest')

    wx2 = convolve(w, k2, mode='nearest')
    wy2 = convolve(w, k2.reshape(1, 3, 1), mode='nearest')
    wz2 = convolve(w, k2.reshape(3, 1, 1), mode='nearest')
             
    wxsq = np.power(wx1, 2) + np.power(convolve(wy2, k3, mode='nearest'), 2) + np.power(convolve(wz2, k3, mode='nearest'), 2)
    wysq = np.power(wy1, 2) + np.power(convolve(wx2, k3.reshape(1, 2, 1), mode='nearest'), 2) + np.power(convolve(wz2, k3.reshape(1, 1, 2), mode='nearest'), 2)
    wzsq = np.power(wz1, 2) + np.power(convolve(wx2, k3.reshape(2, 1, 1), mode='nearest'), 2) + np.power(convolve(wy2, k3.reshape(2, 1, 1), mode='nearest'), 2)

    PsiSmoothEast = psiDeriv(uxsq + vxsq + wxsq)
    PsiSmoothSouth = psiDeriv(uysq + vysq + wysq)
    PsiSmoothBack = psiDeriv(uzsq + vzsq + wzsq)

    PsiSmoothEast[:, -1, :] = 0
    PsiSmoothSouth[-1, :, :] = 0
    PsiSmoothBack[:, :, -1] = 0

    return PsiSmoothEast, PsiSmoothSouth, PsiSmoothBack

def psiDeriv(s2, epsilon = None):
    #-----------------------------------------------------------------------------
    # psiDeriv
    # calculates derivative of psi
    #
    # psiVal = psiDeriv(s2, epsilon) calculates derivative of psi with
    # correction factor to prevent from NaN and Inf
    #
    # Design and implementation by:
    # Tessa Nogatz <tessa.nogatz@itwm.fraunhofer.de>  
    # (c) 2018-2021 Fraunhofer ITWM, Kaiserslautern
    #-----------------------------------------------------------------------------

    if epsilon == None:
        epsilon = 0.001;

    psiVal = 0.5 / np.sqrt( s2 + pow(epsilon, 2))
    return psiVal

def warpImage3D(I1, u, v, w):
    #-----------------------------------------------------------------------------
    # warpImage3D
    # warps image by vector field
    #
    # I1warped = warpImage3D(I1, u, v, w) warps image I1 by given vectorfield
    # [u, v, w] via trilinear interpolation. Sets NaN to zero and outside 
    # coordinates to boundary positions.
    #
    # Design and implementation by:
    # Tessa Nogatz <tessa.nogatz@itwm.fraunhofer.de>  
    # (c) 2018-2021 Fraunhofer ITWM, Kaiserslautern
    #-----------------------------------------------------------------------------

    I1 = I1.astype(float)
    
    u1 = np.copy(u)
    v1 = np.copy(v)
    w1 = np.copy(w)
    
    u1[np.isnan(u1)] = 0
    v1[np.isnan(v1)] = 0
    w1[np.isnan(w1)] = 0
    
    depth = I1.shape[0]
    height = I1.shape[1]
    width = I1.shape[2]

    interpolation_function = RegularGridInterpolator((np.arange(depth), np.arange(height), np.arange(width)), I1)
    
    Z, Y, X = np.meshgrid(np.arange(depth), np.arange(height), np.arange(width), indexing='ij')
    
    xTranslated = np.array(np.clip(X + u1, 0, width - 1))
    yTranslated = np.array(np.clip(Y + v1, 0, height - 1))
    zTranslated = np.array(np.clip(Z + w1, 0, depth - 1))
                           
    xTranslated = xTranslated.flatten()
    yTranslated = yTranslated.flatten()
    zTranslated = zTranslated.flatten()
    
    xyz_list = np.vstack((zTranslated, yTranslated, xTranslated)).T
    
    I1warped = interpolation_function(xyz_list)
    I1warped = I1warped.reshape(I1.shape, order='C')
    
    I1warped[np.isnan(u)|np.isnan(v)|np.isnan(w)] = float("NaN")

    return I1warped

def centralDiffDir(I):
    #-----------------------------------------------------------------------------
    # centralDiffDir
    # calculates central difference five point stencil in all directions
    #
    # [Ix, Iy, Iz] = centralDiffDir(I) calculates the derivative in all
    # coordinate directions of an image I by appriximation via 5 point stencils
    #
    # Design and implementation by:
    # Tessa Nogatz <tessa.nogatz@itwm.fraunhofer.de>  
    #(c) 2018-2021 Fraunhofer ITWM, Kaiserslautern
    #-----------------------------------------------------------------------------

    k2 = (1/12)*np.array([1, -8, 0, 8, -1])
    k2 = k2.reshape(1, 1, 5)

    Ix = convolve(I, k2,  mode='nearest')
    Iy = convolve(I, k2.reshape(1, 5, 1), mode='nearest')
    Iz = convolve(I, k2.reshape(5, 1, 1), mode='nearest')

    return Ix, Iy, Iz

def OpticalFlow3D(original_image, deformed_image, parameters):
    #-----------------------------------------------------------------------------
    # OpticalFlow3D 
    # Multilevel 3D Optical Flow estimation
    #
    # [u,v,w] = OpticalFlow3D(original_image, deformed_image, parameters) calculates the displacement
    # between original_image and deformed_image and stores them in the distinct matrices [u,v,w].
    # Parameters for this calculation are stored in the struct parameters. Note: the
    # images have to be of equal size. If this is not the case, we recommend
    # zero padding to achieve this.
    #
    # Calls for: 
    #            centralDiffDir, warpImage3D, psiDeriv, getPsiSmooth3D, 
    #            successiveOverrelaxation3D
    #           
    # For example use see testFoamScript.
    #
    # Design and matlab implementation by:
    # Tessa Nogatz <tessa.nogatz@itwm.fraunhofer.de>  
    # (c) 2018-2021 Fraunhofer ITWM, Kaiserslautern
    #-----------------------------------------------------------------------------
    
    # Reading the global parameters
    levels = parameters['levels']
    downsampling = parameters['downsampling']
    sigma = parameters['sigma']
    warps = parameters['warps']
    outer_iters = parameters['outer_iters']
    mu = parameters['mu']
    mlambda = parameters['lambda']
    sor_iters = parameters['sor_iters']
    w_param = parameters['w']
    median_filtering = parameters['median_filtering']
    medianx = parameters['medianx']
    mediany = parameters['mediany']
    medianz = parameters['medianz']
    
    # To ensure that the input image type is float
    original_image = original_image.astype(float)
    deformed_image = deformed_image.astype(float)
    
    # Init arrays, these are updated after every iteration
    u0 = np.zeros((np.array(original_image.shape) * pow(downsampling, levels-1)).astype(int))
    v0 = np.zeros((np.array(original_image.shape) * pow(downsampling, levels-1)).astype(int))
    w0 = np.zeros((np.array(original_image.shape) * pow(downsampling, levels-1)).astype(int))

    height = u0.shape[1]
    width = u0.shape[2]
    depth = u0.shape[0]

    # Iterating over levels
    for level in range(levels,1,-1):
        
        print('Current level is ' + str(levels-level+1) + ' / ' + str(levels))
        start = time.time()
        
        I0 = np.copy(original_image)
        I1 = np.copy(deformed_image)
        
        if level > 1:
            new_size = np.round(np.array(original_image.shape) * pow(downsampling, level-1))
            I0 = resize(I0, new_size.astype(int), anti_aliasing = True, anti_aliasing_sigma = sigma*level)
            I1 = resize(I1, new_size.astype(int), anti_aliasing = True, anti_aliasing_sigma = sigma*level)
            
       
        for warp in range(1, warps):
            # compute the derivatives, smoothing considering the directions
            Ix0, Iy0, Iz0 = centralDiffDir(I0) # todo
            Ix1, Iy1, Iz1 = centralDiffDir(I1) # todo
        
            # warp the derivative according to the current displacement
            Ix1_warped = warpImage3D(Ix1, u0, v0, w0) # todo
            Iy1_warped = warpImage3D(Iy1, u0, v0, w0) # todo
            Iz1_warped = warpImage3D(Iz1, u0, v0, w0) # todo
        
            # estimate derivative direction by mixing i1 and i2 derivatives
            Ix_warped = 0.5 * (Ix1_warped + Ix0) 
            Iy_warped = 0.5 * (Iy1_warped + Iy0) 
            Iz_warped = 0.5 * (Iz1_warped + Iz0) 
        
            I1_warped = warpImage3D(I1, u0, v0, w0) # todo
            Id_warped = I1_warped-I0
        
            del I1_warped
        
            # second order derivatives
            Ixx0, Ixy0, Ixz0 = centralDiffDir(Ix0) # todo
            _,    Iyy0, Iyz0 = centralDiffDir(Iy0) # todo
            _,       _, Izz0 = centralDiffDir(Iz0) # todo
        
            Ixd_warped = Ix1_warped - Ix0
            Iyd_warped = Iy1_warped - Iy0
            Izd_warped = Iz1_warped - Iz0
            
            del Ix0, Iy0, Iz0, Ix1_warped, Iy1_warped, Iz1_warped
        
            Ixx1, Ixy1, Ixz1 = centralDiffDir(Ix1) # todo
            _,    Iyy1, Iyz1 = centralDiffDir(Iy1) # todo
            _,       _, Izz1 = centralDiffDir(Iz1) # todo 
            
            del Ix1, Iy1, Iz1
                
            # warp second order derivatives
            Ixx_warped = 0.5 * (warpImage3D(Ixx1, u0, v0, w0) + Ixx0)
            del Ixx0, Ixx1
            Ixy_warped = 0.5 * (warpImage3D(Ixy1, u0, v0, w0) + Ixy0)
            del Ixy0, Ixy1
            Ixz_warped = 0.5 * (warpImage3D(Ixz1, u0, v0, w0) + Ixz0)
            del Ixz0, Ixz1
            Iyy_warped = 0.5 * (warpImage3D(Iyy1, u0, v0, w0) + Iyy0)
            del Iyy0, Iyy1
            Iyz_warped = 0.5 * (warpImage3D(Iyz1, u0, v0, w0) + Iyz0)
            del Iyz0, Iyz1
            Izz_warped = 0.5 * (warpImage3D(Izz1, u0, v0, w0) + Izz0)
            del Izz0, Izz1
        
            du = np.zeros(I0.shape)
            dv = np.zeros(I0.shape)
            dw = np.zeros(I0.shape)
        
            for outer_iter in range(1, outer_iters):
                print("Iter " + str(outer_iter) + " warp " + str(warp))
                
                part1 = np.power(Id_warped + Ix_warped * du + Iy_warped * dv + Iz_warped * dw, 2)
                part2 = mu * np.power((Ixd_warped + Ixx_warped * du + Ixy_warped * dv + Ixz_warped * dw), 2)
                part3 = np.power(Iyd_warped + Ixy_warped * du + Iyy_warped * dv + Iyz_warped * dw, 2)
                part4 = np.power((Izd_warped + Ixz_warped * du + Iyz_warped * dv + Izz_warped * dw), 2)
                
                PsiData = psiDeriv(part1 + part2 + part3 + part4)

                PsiSmoothEast, PsiSmoothSouth, PsiSmoothBack = getPsiSmooth3D(u0 + du, v0 + dv, w0 + dw)
                relaxation_start = time.time()
                successiveOverrelaxation3D( \
                    Ix_warped.ravel(order='F'), Iy_warped.ravel(order='F'), Iz_warped.ravel(order='F'), Id_warped.ravel(order='F'), \
                    Ixx_warped.ravel(order='F'), Ixy_warped.ravel(order='F'), Ixz_warped.ravel(order='F'), \
                    Iyy_warped.ravel(order='F'), Iyz_warped.ravel(order='F'), Izz_warped.ravel(order='F'), \
                    Ixd_warped.ravel(order='F'), Iyd_warped.ravel(order='F'), Izd_warped.ravel(order='F'), \
                    PsiData.ravel(order='F'), PsiSmoothEast.ravel(order='F'), PsiSmoothSouth.ravel(order='F'), PsiSmoothBack.ravel(order='F'), \
                    du.ravel(order='F'), dv.ravel(order='F'), dw.ravel(order='F'), u0.ravel(order='F'), v0.ravel(order='F'), w0.ravel(order='F'), \
                    I0.shape[0], I0.shape[1], I0.shape[2], mlambda, mu, \
                    sor_iters, w_param)
                relaxation_end = time.time()
                print("Relaxation " + str(relaxation_end - relaxation_start) + " s")
            
            
                if outer_iter == outer_iters and median_filtering:
                    du = median_filter(du, (medianx, mediany, medianz))
                    dv = median_filter(dv, (medianx, mediany, medianz))
                    dw = median_filter(dw, (medianx, mediany, medianz))

            # update the optical flow
            u = u0 + du
            v = v0 + dv
            w = w0 + dw
            
            Path("plots").mkdir(parents=True, exist_ok=True)
            io.imsave("plots/u_level" + str(level) + ".tif", u[:, int(u.shape[1]/2), :].astype(np.float32))
            io.imsave("plots/v_level" + str(level) + ".tif", v[int(v.shape[0]/2), :, :].astype(np.float32))
            io.imsave("plots/w_level" + str(level) + ".tif", w[:, :, int(w.shape[1]/2)].astype(np.float32))
            
        print("Min du: " + str(np.amin(du)) + " max du: " + str(np.amax(du)) + " min dv: " + str(np.amin(dv)) + " max dv: " + str(np.amax(dv)) + " min dw: " + str(np.amin(dw)) + " max dw: " + str(np.amax(dw)))
        
        if level > 1:
           # interpolate to get the initial value of the finner pyramid level
           new_size = np.round(np.array(original_image.shape) * pow(downsampling, level-2))
           u0 = resize(u0, new_size.astype(int), anti_aliasing = False)
           v0 = resize(v0, new_size.astype(int), anti_aliasing = False)
           w0 = resize(w0, new_size.astype(int), anti_aliasing = False)
           
                      
        del u, v, w
        del Ix_warped, Iy_warped, Iz_warped, Id_warped, Ixx_warped, Ixy_warped, Ixz_warped, Iyy_warped, Iyz_warped, Izz_warped, Ixd_warped, Iyd_warped, Izd_warped, PsiData, PsiSmoothEast, PsiSmoothSouth, PsiSmoothBack
        
        end = time.time()

        print("Iteration done - processing took " + str(end - start) + " s")
