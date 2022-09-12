import cython
import numpy as np
from tqdm import trange

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.initializedcheck(False)   

cpdef successiveOverrelaxation3D(double[::1] ix, double[::1] iy, double[::1] iz, double[::1] id, double[::1] ixx, double[::1] ixy, double[::1] ixz, double[::1] iyy, double[::1] iyz, double[::1] izz, double[::1] ixd, double[::1] iyd, double[::1] izd, double[::1] psi_data, double[::1] psi_smooth_east, double[::1] psi_smooth_south, double[::1] psi_smooth_back, double[::1] du, double[::1] dv, double[::1] dw, double[::1] u, double[::1] v, double[::1] w, int depth, int height, int width, \
    double mlambda, double mu, int sor_iter, double wrelax):

    cdef int iter
    cdef int i, a, b, c, d
	cdef unsigned long long int n
    cdef double psi_n, psi_s, psi_w, psi_e, psi_f, psi_b, u_n, u_s, u_w, u_e, u_f, u_b, v_n, v_s, v_w, v_e, v_f, v_b, w_n, w_s, w_w, w_e, w_f, w_b, du_n, du_s, du_w, du_e, du_f, du_b, dv_n, dv_s, dv_w, dv_e, dv_f, dv_b, dw_n, dw_s, dw_w, dw_e, dw_f, dw_b, b1DivPart, b2DivPart, b3DivPart
    cdef double B_coef_u1, B_coef_u2, B_const_u, B_coef_v1, B_coef_v2, B_const_v, B_coef_w1, B_coef_w2, B_const_w#, Aiipart_u, Aiipart_v, Aiipart_w
    cdef double wrelax_per_lambda, psi_parts1, psi_parts2
	
	#original:  p,     q,      r
	#modified:depth, height, width
	#Axis :     z      y       x
	
    n = depth * height * width
    
    #cdef double[:] B_coef_u1 = np.empty(n, dtype=np.double)
    #cdef double[:] B_coef_u2 = np.empty(n, dtype=np.double)
    #cdef double[:] B_const_u = np.empty(n, dtype=np.double)
    #cdef double[:] B_coef_v1 = np.empty(n, dtype=np.double)
    #cdef double[:] B_coef_v2 = np.empty(n, dtype=np.double)
    #cdef double[:] B_const_v = np.empty(n, dtype=np.double)
    #cdef double[:] B_coef_w1 = np.empty(n, dtype=np.double)
    #cdef double[:] B_coef_w2 = np.empty(n, dtype=np.double)
    #cdef double[:] B_const_w = np.empty(n, dtype=np.double)
    cdef double[:] Aiipart_u = np.empty(n, dtype=np.double)
    cdef double[:] Aiipart_v = np.empty(n, dtype=np.double)
    cdef double[:] Aiipart_w = np.empty(n, dtype=np.double)
    wrelax_per_lambda = wrelax / mlambda
   
    for i in range(n):
    #    B_coef_u1[i] = psi_data[i] * (ix[i] * iy[i] + mu * ixx[i] * ixy[i] + mu * ixy[i] * iyy[i] + mu * ixz[i] * iyz[i])
    #    B_coef_u2[i] = psi_data[i] * (ix[i] * iz[i] + mu * ixx[i] * ixz[i] + mu * ixy[i] * iyz[i] + mu * ixz[i] * izz[i])
    #    B_const_u[i] = psi_data[i] * (ix[i] * id[i] + mu * ixx[i] * ixd[i] + mu * ixy[i] * iyd[i] + mu * ixz[i] * izd[i]) 
    #    B_coef_v1[i] = psi_data[i] * (ix[i] * iy[i] + mu * ixx[i] * ixy[i] + mu * ixy[i] * iyy[i] + mu * iyz[i] * ixz[i])   
    #    B_coef_v2[i] = psi_data[i] * (iy[i] * iz[i] + mu * ixy[i] * ixz[i] + mu * iyy[i] * iyz[i] + mu * iyz[i] * izz[i])
    #    B_const_v[i] = psi_data[i] * (iy[i] * id[i] + mu * ixy[i] * ixd[i] + mu * iyy[i] * iyd[i] + mu * iyz[i] * izd[i]) 
    #    B_coef_w1[i] = psi_data[i] * (ix[i] * iz[i] + mu * ixx[i] * ixz[i] + mu * ixy[i] * iyz[i] + mu * ixz[i] * izz[i])
    #    B_coef_w2[i] = psi_data[i] * (iy[i] * iz[i] + mu * ixy[i] * ixz[i] + mu * iyy[i] * iyz[i] + mu * iyz[i] * izz[i])
    #    B_const_w[i] = psi_data[i] * (iz[i] * id[i] + mu * ixz[i] * ixd[i] + mu * iyz[i] * iyd[i] + mu * izz[i] * izd[i]) 
        Aiipart_u[i] = psi_data[i] * (pow(ix[i], 2) + mu * pow(ixx[i], 2) + mu * pow(ixy[i], 2) + mu * pow(ixz[i], 2))
        Aiipart_v[i] = psi_data[i] * (pow(iy[i], 2) + mu * pow(ixy[i], 2) + mu * pow(iyy[i], 2) + mu * pow(iyz[i], 2)) 
        Aiipart_w[i] = psi_data[i] * (pow(iz[i], 2) + mu * pow(ixz[i], 2) + mu * pow(iyz[i], 2) + mu * pow(izz[i], 2)) 

    for iter in range(sor_iter):
    
        #print("Iter " + str(iter))
        
        for i in range(n):
            
            B_coef_u1 = psi_data[i] * (ix[i] * iy[i] + mu * ixx[i] * ixy[i] + mu * ixy[i] * iyy[i] + mu * ixz[i] * iyz[i])
            B_coef_u2 = psi_data[i] * (ix[i] * iz[i] + mu * ixx[i] * ixz[i] + mu * ixy[i] * iyz[i] + mu * ixz[i] * izz[i])
            B_const_u = psi_data[i] * (ix[i] * id[i] + mu * ixx[i] * ixd[i] + mu * ixy[i] * iyd[i] + mu * ixz[i] * izd[i]) 
            B_coef_v1 = psi_data[i] * (ix[i] * iy[i] + mu * ixx[i] * ixy[i] + mu * ixy[i] * iyy[i] + mu * iyz[i] * ixz[i])   
            B_coef_v2 = psi_data[i] * (iy[i] * iz[i] + mu * ixy[i] * ixz[i] + mu * iyy[i] * iyz[i] + mu * iyz[i] * izz[i])
            B_const_v = psi_data[i] * (iy[i] * id[i] + mu * ixy[i] * ixd[i] + mu * iyy[i] * iyd[i] + mu * iyz[i] * izd[i]) 
            B_coef_w1 = psi_data[i] * (ix[i] * iz[i] + mu * ixx[i] * ixz[i] + mu * ixy[i] * iyz[i] + mu * ixz[i] * izz[i])
            B_coef_w2 = psi_data[i] * (iy[i] * iz[i] + mu * ixy[i] * ixz[i] + mu * iyy[i] * iyz[i] + mu * iyz[i] * izz[i])
            B_const_w = psi_data[i] * (iz[i] * id[i] + mu * ixz[i] * ixd[i] + mu * iyz[i] * iyd[i] + mu * izz[i] * izd[i]) 
            #Aiipart_u = psi_data[i] * (pow(ix[i], 2) + mu * pow(ixx[i], 2) + mu * pow(ixy[i], 2) + mu * pow(ixz[i], 2))
            #Aiipart_v = psi_data[i] * (pow(iy[i], 2) + mu * pow(ixy[i], 2) + mu * pow(iyy[i], 2) + mu * pow(iyz[i], 2)) 
            #Aiipart_w = psi_data[i] * (pow(iz[i], 2) + mu * pow(ixz[i], 2) + mu * pow(iyz[i], 2) + mu * pow(izz[i], 2)) 
        
		    #original:  p,     q,      r
			#modified:depth, height, width
			#Axis :     z      y       x
		
            d = i % (depth * height)
            a = (i - d) / (depth * height)
            b = d % depth
            c = (d - b) / depth
            
            psi_n = 0.0 if b == 0 else psi_smooth_south[i-1]
            psi_s = 0.0 if b == depth-1 else psi_smooth_south[i]
            psi_w = 0.0 if c == 0 else psi_smooth_east[i-depth]
            psi_e = 0.0 if c == height-1 else psi_smooth_east[i]
            psi_f = 0.0 if a == 0 else psi_smooth_back[i-depth*height]
            psi_b = 0.0 if a == width-1 else psi_smooth_back[i]
            
            u_n = 0.0 if b == 0 else u[i-1]
            u_s = 0.0 if b == depth-1 else u[i+1]
            u_w = 0.0 if c == 0 else u[i-depth]
            u_e = 0.0 if c == height-1 else u[i+depth]
            u_f = 0.0 if a == 0 else u[i-depth*height]
            u_b = 0.0 if a == width-1 else u[i+height*depth]
            
            v_n = 0.0 if b == 0 else v[i-1]
            v_s = 0.0 if b == depth-1 else v[i+1]
            v_w = 0.0 if c == 0 else v[i-depth]
            v_e = 0.0 if c == height-1 else v[i+depth]
            v_f = 0.0 if a == 0 else v[i-depth*height]
            v_b = 0.0 if a == width-1 else v[i+height*depth]
            
            w_n = 0.0 if b == 0 else w[i-1]
            w_s = 0.0 if b == depth-1 else w[i+1]
            w_w = 0.0 if c == 0 else w[i-depth]
            w_e = 0.0 if c == height-1 else w[i+depth]
            w_f = 0.0 if a == 0 else w[i-depth*height]
            w_b = 0.0 if a == width-1 else w[i+height*depth]
            
            du_n = 0.0 if b == 0 else du[i-1]
            du_s = 0.0 if b == depth-1 else du[i+1]
            du_w = 0.0 if c == 0 else du[i-depth]
            du_e = 0.0 if c == height-1 else du[i+depth]
            du_f = 0.0 if a == 0 else du[i-depth*height]
            du_b = 0.0 if a == width-1 else du[i+height*depth]
            
            dv_n = 0.0 if b == 0 else dv[i-1]
            dv_s = 0.0 if b == depth-1 else dv[i+1]
            dv_w = 0.0 if c == 0 else dv[i-depth]
            dv_e = 0.0 if c == height-1 else dv[i+depth]
            dv_f = 0.0 if a == 0 else dv[i-depth*height]
            dv_b = 0.0 if a == width-1 else dv[i+height*depth]
            
            dw_n = 0.0 if b == 0 else dw[i-1]
            dw_s = 0.0 if b == depth-1 else dw[i+1]
            dw_w = 0.0 if c == 0 else dw[i-depth]
            dw_e = 0.0 if c == height-1 else dw[i+depth]
            dw_f = 0.0 if a == 0 else dw[i-depth*height]
            dw_b = 0.0 if a == width-1 else dw[i+height*depth]

            b1DivPart = psi_n * (u_n + du_n) + psi_s * (u_s + du_s) + psi_e * (u_e + du_e) + psi_w * (u_w + du_w) + psi_f * (u_f + du_f) + psi_b * (u_b + du_b) 
            b2DivPart = psi_n * (v_n + dv_n) + psi_s * (v_s + dv_s) + psi_e * (v_e + dv_e) + psi_w * (v_w + dv_w) + psi_f * (v_f + dv_f) + psi_b * (v_b + dv_b)
            b3DivPart = psi_n * (w_n + dw_n) + psi_s * (w_s + dw_s) + psi_e * (w_e + dw_e) + psi_w * (w_w + dw_w) + psi_f * (w_f + dw_f) + psi_b * (w_b + dw_b)
            
            psi_parts1 = psi_n + psi_e + psi_w + psi_s + psi_b + psi_f
            psi_parts2 = psi_n + psi_e + psi_w + psi_s + psi_f + psi_b
            
            try:
                du[i] = (1 - wrelax)*du[i] + (wrelax*(b1DivPart - psi_parts1*u[i]) - wrelax_per_lambda*(B_coef_u1*dv[i] + B_coef_u2*dw[i] + B_const_u)) / (psi_parts2 + (1/mlambda)*Aiipart_u[i])
                dv[i] = (1 - wrelax)*dv[i] + (wrelax*(b2DivPart - psi_parts1*v[i]) - wrelax_per_lambda*(B_coef_v1*du[i] + B_coef_v2*dw[i] + B_const_v)) / (psi_parts2 + (1/mlambda)*Aiipart_v[i])
                dw[i] = (1 - wrelax)*dw[i] + (wrelax*(b3DivPart - psi_parts1*w[i]) - wrelax_per_lambda*(B_coef_w1*du[i] + B_coef_w2*dv[i] + B_const_w)) / (psi_parts2 + (1/mlambda)*Aiipart_w[i])
            except:
                du[i] = 0
                dv[i] = 0
                dw[i] = 0
                print("Division error (psi_parts2 + (1/mlambda)*Aiipart_u[i]) = " + str(psi_parts2 + (1/mlambda)*Aiipart_u[i]))
    return




