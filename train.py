import torch.backends.cudnn as cudnn
cudnn.benchmark=False

import numpy as np
import time
import os
import lpips
from data import data_loader as dl
import argparse
from util.visualizer import Visualizer
from IPython import embed
from Test_TestSet import Test_TestSet
import csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', default=['./dataset/TexturedDB_80%_TrainList_withnbPatchesPerVP_threth0.6.csv'], help='datasets to train on')
    parser.add_argument('--model', type=str, default='lpips', help='distance model type [lpips] for linearly calibrated net, [baseline] for off-the-shelf network, [l2] for euclidean distance, [ssim] for Structured Similarity Image Metric')
    parser.add_argument('--net', type=str, default='alex', help='[squeeze], [alex], or [vgg] for network architectures')
    #parser.add_argument('--batch_size', type=int, default=50, help='batch size to test image patches in')
    parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='gpus to use')

    parser.add_argument('--nThreads', type=int, default=10, help='number of threads to use in data loader')
    parser.add_argument('--nepoch', type=int, default=5, help='# epochs at base learning rate')
    parser.add_argument('--nepoch_decay', type=int, default=5, help='# additional epochs at linearly learning rate')
    parser.add_argument('--npatches', type=int, default=65, help='# randomly sampled image patches')
    parser.add_argument('--nInputImg', type=int, default=4, help='# stimuli/images in each batch')
    parser.add_argument('--lr', type=float, default=0.0001, help='# initial learning rate')
    
    parser.add_argument('--testset_freq', type=int, default=5, help='frequency of evaluating the testset')
    parser.add_argument('--display_freq', type=int, default=50000, help='frequency (in instances) of showing training results on screen')
    parser.add_argument('--print_freq', type=int, default=50000, help='frequency (in instances) of showing training results on console')
    parser.add_argument('--save_latest_freq', type=int, default=20000, help='frequency (in instances) of saving the latest results')
    parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--display_id', type=int, default=0, help='window id of the visdom display, [0] for no displaying')
    parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
    parser.add_argument('--display_port', type=int, default=8001,  help='visdom display port')
    parser.add_argument('--use_html', action='store_true', help='save off html pages')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='checkpoints directory')
    parser.add_argument('--name', type=str, default='tmp', help='directory name for training')

    parser.add_argument('--from_scratch', action='store_true', help='model was initialized from scratch')
    parser.add_argument('--train_trunk', action='store_true', help='model trunk was trained/tuned')
    parser.add_argument('--train_plot', action='store_true', help='plot saving')

    opt = parser.parse_args()
    opt.batch_size = opt.npatches * opt.nInputImg
    
    opt.save_dir = os.path.join(opt.checkpoints_dir,opt.name)
    if(not os.path.exists(opt.save_dir)):
        os.mkdir(opt.save_dir)

    # initialize model
    trainer = lpips.Trainer()
    trainer.initialize(model=opt.model, net=opt.net, use_gpu=opt.use_gpu, is_train=True, lr=opt.lr,
        pnet_rand=opt.from_scratch, pnet_tune=opt.train_trunk, gpu_ids=opt.gpu_ids)

    load_size = 64 # default value is 64

    visualizer = Visualizer(opt)

    # load data from all test sets 
    # The random patches for the test set are only sampled once at the beginning of training in order to avoid noise in the validation loss.
    Testset = './dataset/TexturedDB_20%_TestList_withnbPatchesPerVP_threth0.6.csv'
    data_loader_testSet = dl.CreateDataLoader(Testset,dataset_mode='2afc', Nbpatches= opt.npatches, 
                                              load_size = load_size, batch_size=opt.batch_size, nThreads=opt.nThreads)
    test_TestSet = Test_TestSet(opt)
    total_steps = 0
    # fid = open(os.path.join(opt.checkpoints_dir,opt.name,'train_log.txt'),'w+')
    # f_hyperParam = open(os.path.join(opt.checkpoints_dir,opt.name,'tuning_hyperparam.csv'),'a') 
    # if os.stat(os.path.join(opt.checkpoints_dir,opt.name,'tuning_hyperparam.csv')).st_size == 0:
        # f_hyperParam.write("nepoch,nepoch_decay,npatches,nInputImg,lr,epoch,TrainLoss,testLoss,SROCC_testset\n")
    
    start_time = time.time()
    for epoch in range(1, opt.nepoch + opt.nepoch_decay + 1):
            # Load training data to sample random patches every epoch
            data_loader = dl.CreateDataLoader(opt.datasets,dataset_mode='2afc', trainset=True, Nbpatches=opt.npatches, 
                                              load_size = load_size, batch_size=opt.batch_size, serial_batches=True, nThreads=opt.nThreads)
            dataset = data_loader.load_data()
            dataset_size = len(data_loader)
            D = len(dataset)

            epoch_start_time = time.time()
            nb_batches = 0 
            Loss_trainset = 0 
            for i, data in enumerate(dataset):
                iter_start_time = time.time()
                total_steps += opt.batch_size
                epoch_iter = total_steps - dataset_size * (epoch - 1)

                trainer.set_input(data)
                trainer.optimize_parameters()

                # if total_steps % opt.display_freq == 0:
                    # visualizer.display_current_results(trainer.get_current_visuals(), epoch)

                errors = trainer.get_current_errors() # current error per batch
                Loss_trainset += errors['loss_total'] # total loss over trainset = sum(Loss/batch)/nb_batches
                nb_batches += 1 

                # if total_steps % opt.print_freq == 0:
                    # t = (time.time()-iter_start_time)/opt.batch_size
                    # t2o = (time.time()-epoch_start_time)/3600.
                    # t2 = t2o*D/(i+.0001)
                    # visualizer.print_current_errors(epoch, epoch_iter, errors, t, t2=t2, t2o=t2o, fid=fid)

                    #for key in errors.keys():
                        #visualizer.plot_current_errors_save(epoch, float(epoch_iter)/dataset_size, opt, errors, keys=[key,], name=key, to_plot=opt.train_plot)

                    # if opt.display_id > 0:
                        # visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

                # if total_steps % opt.save_latest_freq == 0:
                    # print('saving the latest model (epoch %d, total_steps %d)' %(epoch, total_steps))
                    # trainer.save(opt.save_dir, 'latest')

            if epoch % opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' %
                      (epoch, total_steps))
                trainer.save(opt.save_dir, 'latest')
                trainer.save(opt.save_dir, epoch)
                
                print('nb batch %.1f'%nb_batches)
                Loss_trainset = Loss_trainset/nb_batches
                print('Epoch Loss %.6f'%Loss_trainset)
                resPerEpoch = dict([('Trainset_Totalloss', Loss_trainset)])
                
                for key in resPerEpoch.keys():
                    visualizer.plot_current_errors_save(epoch, float(0), opt, resPerEpoch, keys=[key,], name=key, to_plot=opt.train_plot)


            # Evaluate the Test set at the End of the epoch
            if epoch % opt.testset_freq == 0:
                res_testset = lpips.Testset_DSIS(data_loader_testSet, trainer.forward, trainer.rankLoss.forward, name=Testset) # SROCC & loss
                for Tkey in res_testset.keys():
                    test_TestSet.plot_TestSet_save(epoch=epoch, res=res_testset, keys=[Tkey,],  name=Tkey, to_plot=opt.train_plot, what_to_plot = 'TestSet_Res')
                info = str(opt.nepoch) + "," + str(opt.nepoch_decay) + "," + str(opt.npatches) + "," + str(opt.nInputImg) + "," + str(opt.lr) + "," + str(epoch) + "," + str(Loss_trainset) + "," + str(res_testset['loss']) + "," + str(res_testset['SROCC']) + "\n"
            else:
                info = str(opt.nepoch) + "," + str(opt.nepoch_decay) + "," + str(opt.npatches) + "," + str(opt.nInputImg) + "," + str(opt.lr) + "," + str(epoch) + "," + str(Loss_trainset) + "\n"
            
            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, opt.nepoch + opt.nepoch_decay, time.time() - epoch_start_time))

            #f_hyperParam.write(info)
            
            if epoch > opt.nepoch:
                trainer.update_learning_rate(opt.nepoch_decay)

    # trainer.save_done(True)
    # fid.close()
    #f_hyperParam.close()
    print( 'End of %d epochs. Time taken: %d sec' %(opt.nepoch + opt.nepoch_decay,  time.time() -  start_time))
    
