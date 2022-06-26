import numpy as np
import platform
import os
import nibabel


if (platform.system() == "Windows") :
    path = os.environ['USERPROFILE'] + '\Desktop'
else :
    path = '/home/yy3/vemuri/new_data'

bvals = np.loadtxt(os.path.join(path,'example_file','3104_BL_bvals'))
bvecs = np.loadtxt(os.path.join(path,'example_file','3104_BL_bvecs'))
is_b0 = ( abs(bvals) < 0.0001 )
is_b1000 = ( abs(bvals-1000) < 0.0001 )
svecs = bvecs[:,is_b1000]
data = nibabel.load(os.path.join(path,'example_file','3104_BL_data_subject_space.nii.gz'))
ndata = np.array(data.dataobj)

b0data = ndata[:,:,:,is_b0]
b1000data = ndata[:,:,:,is_b1000]
