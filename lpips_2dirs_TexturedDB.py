import argparse
import os
import lpips
import torch
import numpy as np
import statsmodels.api as sm
from scipy import stats

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='./dataset/PlaylistsStimuli_patches_withVP_threth0.6')
parser.add_argument('-d1','--dir1', type=str, default='./dataset/References_patches_withVP_threth0.6')
parser.add_argument('-o','--out', type=str, default='./GrpahicsLPIPS_Playlist.csv')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()


## Initializing the model
Network_path = './checkpoints/Trial1/10_net_.pth' # Change to the selected trained network
loss_fn = lpips.LPIPS(net='alex',version=opt.version , model_path = Network_path)
if(opt.use_gpu):
	loss_fn.cuda()
    
## Outout file
f = open(opt.out,'w')
f.writelines('p0,lpips_alex,\n')
files = os.listdir(opt.dir0)

stimuli_patches = []
lpipsscore = []


# crawl directories
for file in files:
	if(os.path.exists(os.path.join(opt.dir0,file))):
		model = file.split("_simpL")[0]
		patchNb = file.rsplit("_P", 1)[1] #=> 1.png
		ref = model + '_Ref_P' + patchNb
    
        # Load images
		img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0,file))) # RGB image from [-1,1]
		img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1,ref)))
		
		if(opt.use_gpu):
			img0 = img0.cuda()
			img1 = img1.cuda()

		# Compute distance
		dist01 = loss_fn.forward(img0,img1).reshape(1,)
		stimuli_patches.append(file.split(".png")[0])
		lpipsscore.append(dist01[0].item())
		if dist01 > 1:
		    dist01 = 1
		            
		f.writelines('%s, %.6f\n'%(file.split(".png")[0],dist01))

lpipsscore = np.array(lpipsscore)
stimuli_patches = np.array(stimuli_patches)

f.close()
