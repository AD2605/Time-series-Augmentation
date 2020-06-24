import os
import random

import numpy
import torch
from torch.utils.data import DataLoader

def get_dataloader(path, batch):
    os.chdir(path)
    train_data = []

    classes = {'769': int(0), '770': int(1), '771': int(2), '772': int(3)}
    folders = ['769', '770', '771', '772']

    for folder in folders:
        files = os.listdir(path + '/' + folder)
        os.chdir(path + '/' + folder)
        for file in files:
            try:
                data = numpy.load(file)
                train_data.append([torch.tensor(data, dtype=torch.float32),torch.tensor(data=numpy.array(classes[folder]), dtype=torch.int64)])

            except Exception as e:
                pass


    data_loader = DataLoader(train_data, batch_size=batch, shuffle=True)
    return data_loader
