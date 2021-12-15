def CreateDataLoader(InputData,dataroot='./dataset',dataset_mode='2afc', trainset=False, Nbpatches= 205, load_size=64,batch_size=1,serial_batches=True,nThreads=4):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    #data_loader.initialize(InputData,dataroot=dataroot+'/'+dataset_mode,dataset_mode=dataset_mode,load_size=load_size,batch_size=batch_size,serial_batches=serial_batches, nThreads=nThreads)
    data_loader.initialize(data_csvfile=InputData, trainset=trainset, Nbpatches=Nbpatches, dataset_mode=dataset_mode,load_size=load_size,batch_size=batch_size,serial_batches=serial_batches, nThreads=nThreads)
    return data_loader
