import argparse
import os
import lpips
import torch
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
from util import util
from itertools import groupby
from operator import itemgetter
from statistics import mean
import csv

class Test_TestSet():
    def __init__(self, opt):
        self.testset_dir = os.path.join(opt.checkpoints_dir, opt.name, 'TestSet')
        #self.Lpips_dir = os.path.join(opt.checkpoints_dir, opt.name, 'Lpips_res')
        self.outfilepath = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs([self.testset_dir,])
        util.mkdirs([self.Lpips_dir,])
        
    def Lpips_TestSet(self, epoch, Net, ModelPath, use_gpu, TestSet_csvFile):
        # Initializing the model
        loss_fn = lpips.LPIPS(net=Net,version='0.1', model_path = ModelPath)
        if(use_gpu):
            loss_fn.cuda()
        
        # crawl csv file
        IMG0 = []
        IMG1 = []
        Judge = []
        with open(TestSet_csvFile) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else: # patches to load
                    IMG0.append(row[0])
                    IMG1.append(row[1])
                    Judge.append(float(row[2]))

        stimuli = []
        lpipsscorePatches = []
        subjscore = []
        
        # output file 
        out = self.outfilepath + '/res_testset_epoch'+str(epoch)+'.csv'
        f = open(out,'w')
        f.writelines('p0,lpips_alex,judge\n')
        
        print('nb patches = %.1f'%len(IMG0))

        for i in range(0, len(IMG0)):
            # Load images
            img0 = lpips.im2tensor(lpips.load_image(IMG0[i])) # RGB image from [-1,1]
            img1 = lpips.im2tensor(lpips.load_image(IMG1[i]))
            
            if(use_gpu):
                with torch.no_grad():
                    img0 = img0.cuda()
                    img1 = img1.cuda()

            # Compute distance
            dist01 = loss_fn.forward(img0,img1).reshape(1,)
            stimulusPatch = (IMG0[i]).split("/PlaylistsStimuli_patches_withVP_threth0.6/")[1] 
            stimulusName = stimulusPatch.rsplit("_P", 1)[0]# split according to the last occurence of "_P"
            stimuli.append(stimulusName)
            lpipsscorePatches.append(dist01[0].item())
            subjscore.append(Judge[i])#subjscore.append(judge.item())
                
            f.writelines('%s, %.6f, %.3f\n'%(stimulusPatch,dist01,Judge[i]))

        f.close()
        # Aggregate patches
        mos = [mean(map(itemgetter(1), group))
            for key, group in groupby(zip(stimuli, subjscore), key=itemgetter(0))]
        
        lpipsscore = [mean(map(itemgetter(1), group))
            for key, group in groupby(zip(stimuli, lpipsscorePatches), key=itemgetter(0))]
        
        lpipsscore = np.array(lpipsscore)
        mos = np.array(mos)
        
        # Compte total loss (MSE)
        loss = (lpipsscore-mos)*(lpipsscore-mos)
        print('Nb Validation stimuli: %.1f'%len(loss))
        loss = np.mean(loss)
        print('Lpips MSE %.3f' %loss)
        
        # Compute correlation
        lpipsscore = sm.add_constant(lpipsscore)
        glm_binom = sm.GLM(mos, lpipsscore, family = sm.families.Binomial())#, link = sm.families.links.Logit()
        res = glm_binom.fit()

        fittedLpips = res.predict()
        corrPears =  stats.pearsonr(fittedLpips, mos)[0]
        corrSpear =  stats.spearmanr(fittedLpips, mos)[0]
        print('Lpips pearson %.3f'%corrPears)
        print('Lpips spearman %.3f'%corrSpear)
        
        resDict = dict([('MSE', loss),
                        ('PLCC', corrPears),
                        ('SROCC', corrSpear)])
        
        return resDict

    def plot_TestSet_save(self, epoch, res, keys='+ALL', name='MSE', to_plot=False, what_to_plot = ''):
        if(what_to_plot == 'Lpips_Res'):
            plot_dir = self.Lpips_dir
        else: 
            plot_dir = self.testset_dir
            
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X':[],'Y':[], 'legend':list(res.keys())}
        self.plot_data['X'].append(epoch )
        self.plot_data['Y'].append([res[k] for k in self.plot_data['legend']])
        
        if(keys=='+ALL'):
            plot_keys = self.plot_data['legend']
        else:
            plot_keys = keys

        if(to_plot):
            (f,ax) = plt.subplots(1,1)
        for (k,kname) in enumerate(plot_keys):
            kk = np.where(np.array(self.plot_data['legend'])==kname)[0][0]
            x = self.plot_data['X']
            y = np.array(self.plot_data['Y'])[:,kk]
            if(to_plot):
                ax.plot(x, y, 'o-', label=kname)
            np.save(os.path.join(plot_dir,'%s_x')%kname,x)
            np.save(os.path.join(plot_dir,'%s_y')%kname,y)

        if(to_plot):
            plt.legend(loc=0,fontsize='small')
            plt.xlabel('epoch')
            plt.ylabel('Value')
            f.savefig(os.path.join(plot_dir,'%s.png'%name))
            f.clf()
            plt.close()

