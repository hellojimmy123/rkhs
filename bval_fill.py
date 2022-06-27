import numpy as np
import platform
import os
import nibabel
import dipy.core.geometry as geometry
import scipy.interpolate as ipl

PI = np.pi
knot_theta = np.linspace(0, np.pi, num=7)
knot_phi = np.linspace(0, 2*np.pi, num=9)
knot_phi[0] += 0.0001
knot_phi[-1] -= 0.0001
knot_theta[0] += 0.0001
knot_theta[-1] -= 0.0001



if (platform.system() == "Windows") :
    path = os.environ['USERPROFILE'] + '\Desktop'
    path = os.path.join(path,'example_file')
else :
    path = '/home/yy3/Vemuri/3104_BL'

bvals = np.loadtxt(os.path.join(path, '3104_BL_bvals'))
bvecs = np.loadtxt(os.path.join(path, '3104_BL_bvecs'))
is_b0 = ( abs(bvals) < 0.0001 )
is_b1000 = ( abs(bvals-1000) < 0.0001 )
svecs = bvecs[:,is_b1000]
data = nibabel.load(os.path.join(path,'3104_BL_data_subject_space.nii.gz'))
ndata = np.array(data.dataobj)

b0data = ndata[:,:,:,is_b0]
b1000data = ndata[:,:,:,is_b1000]
_,theta,phi = geometry.cart2sphere( svecs[0,:], svecs[1,:], svecs[2,:])
theta,phi = np.mod(theta,np.pi), np.mod(phi,2*np.pi)
# print(np.min(theta)/PI,np.max(theta)/PI)
# print(np.min(phi)/PI,np.max(phi)/PI)
t1, t2 = 10, 10
ftheta, fphi = np.linspace( 0, PI, t1 ), np.linspace( 0, 2*PI, t1 )
# ftheta = np.repeat( ftheta, t2 )
# fphi = np.repeat( fphi, t1 )
#print(phi_grid)

def f(x):
    s1,s2,s3,s4 = x.shape
    fdata = np.zeros((s1,s2,s3,t1,t2))
    print( fdata.nbytes/(1024**2))
    for i in range(s1) :
        print(i)
        for j in range(s2) :
            for k in range(s3) :
                ff = ipl.LSQSphereBivariateSpline(theta, phi, x[i,j,k], knot_theta, knot_phi)
                fdata[i,j,k] = ff(ftheta,fphi)



import time

start = time.time()
f(b1000data)
end = time.time()
print(end - start)

#ff = ipl.LSQSphereBivariateSpline(theta, phi, b1000data[100,100,60], knot_theta, knot_phi)
