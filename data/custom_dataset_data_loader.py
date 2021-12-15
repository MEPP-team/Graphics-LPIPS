import torch.utils.data
from data.base_data_loader import BaseDataLoader
import os

def CreateDataset(dataroots,dataset_mode='2afc',load_size=64, trainset=False , Nbpatches = 205):
    dataset = None
    # Our dataset is baaset on the DSIS protocol (not 2afc). I adapted the code to suit DSIS. However, I did not change the function name.
    if dataset_mode=='2afc': # human judgements
        from data.dataset.twoafc_dataset import TwoAFCDataset
        dataset = TwoAFCDataset()
    elif dataset_mode=='jnd': # human judgements
        from data.dataset.jnd_dataset import JNDDataset
        dataset = JNDDataset()
    else:
        raise ValueError("Dataset Mode [%s] not recognized."%self.dataset_mode)

    dataset.initialize(dataroots,load_size=load_size,Trainset = trainset, maxNbPatches = Nbpatches)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, data_csvfile, trainset=False, Nbpatches=205, dataset_mode='2afc',load_size=64,batch_size=1,serial_batches=True, nThreads=1):
        BaseDataLoader.initialize(self)
        if(not isinstance(data_csvfile,list)):
            data_csvfile = [data_csvfile,]

        self.dataset = CreateDataset(data_csvfile,dataset_mode=dataset_mode,load_size=load_size, trainset=trainset, Nbpatches=Nbpatches)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=not serial_batches,
            num_workers=int(nThreads))
            

    def initialize2(self, datafolders, dataroot='./dataset',dataset_mode='2afc',load_size=64,batch_size=1,serial_batches=True, nThreads=1):
        BaseDataLoader.initialize(self)
        if(not isinstance(datafolders,list)):
            datafolders = [datafolders,]
        data_root_folders = [os.path.join(dataroot,datafolder) for datafolder in datafolders]
        self.dataset = CreateDataset(data_root_folders,dataset_mode=dataset_mode,load_size=load_size)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=not serial_batches,
            num_workers=int(nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
