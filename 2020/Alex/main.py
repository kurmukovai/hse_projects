import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  DataLoader
import torchio as tio

from unet import *


import config
import nibabel
import pandas as pd
import numpy as np

import os, glob

df = pd.DataFrame()
imagepath = []
maskpath = []

for path, _, files in sorted(os.walk(config.data_folder)):
    for file in files:
        if 'flair' in file:
            imagepath.append(path + '/' + file) 
        if 'seg' in file:
            maskpath.append(path + '/' + file) 

# собираем пути 
df['imagepath'] = imagepath[0:350]
df['maskpath'] = maskpath[0:350]
assert len(imagepath) == len(maskpath)
df['dice'] = .0
df['surface_dice'] = .0
# трансформ - к одному классу
def one_class(tensor):
    tensor = tensor != 0
    return tensor

# трансформ - к формату float, ибо модель на выходе выдает именно float
# и дабы все не ломалось, я таргеты также перевожу во float
def to_float(tensor):
    tensor = tensor.type(torch.FloatTensor)
    return tensor

one = tio.Lambda(one_class)


subjects = []
for (image_path, label_path) in zip(df['imagepath'], df['maskpath']):
    subject = tio.Subject(
        image = tio.ScalarImage(image_path),
        mask = tio.LabelMap(label_path)
    )
    subjects.append(subject)

# собираем особый датасет torchio с пациентами
dataset = tio.SubjectsDataset(subjects)

# приводим маску к 1 классу
if config.to_one_class:
    for subject in dataset.dry_iter():
        subject['mask'] = one(subject['mask'])


training_transform = tio.Compose([
    tio.Resample(4), 
    tio.ZNormalization(masking_method=tio.ZNormalization.mean), # вот эту штуку все рекомендовали на форумах torchio. 
    tio.RandomFlip(p=0.25),
    tio.RandomNoise(p=0.25),
    # !!!  Приходится насильно переводить тензоры в float
    tio.Lambda(to_float) 
])

validation_transform = tio.Compose([
    tio.Resample(4),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    tio.RandomNoise(p=0.25),
    tio.Lambda(to_float)
])

def prepare_dataload(patches=True):
    
    training_batch_size = 32
    validation_batch_size = 2 * training_batch_size

    patch_size = 25
    samples_per_volume = 10
    max_queue_length = 300
    sampler = tio.data.UniformSampler(patch_size)

        
    num_subjects = len(dataset)
    num_training_subjects = 245
    num_validation_subjects = 70
    num_test_subjects = 35

    num_split_subjects = num_training_subjects, num_validation_subjects, num_test_subjects
    training_subjects, validation_subjects, test_subjects = torch.utils.data.random_split(dataset, num_split_subjects)

    
    training_set = tio.SubjectsDataset(
        training_subjects, transform=training_transform)

    validation_set = tio.SubjectsDataset(
        validation_subjects, transform=validation_transform)
    
    test_set = tio.SubjectsDataset(
        test_subjects, transform=validation_transform)
    

    patches_training_set = tio.Queue(
        subjects_dataset=training_set,
        max_length=max_queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=2,
        shuffle_subjects=True,
        shuffle_patches=True,
        )

    patches_validation_set = tio.Queue(
        subjects_dataset=validation_set,
        max_length=max_queue_length,
        samples_per_volume=samples_per_volume * 2,
        sampler=sampler,
        num_workers=2,
        shuffle_subjects=False,
        shuffle_patches=False,
        )

    patches_test_set = tio.Queue(
        subjects_dataset=test_set,
        max_length=max_queue_length,
        samples_per_volume=samples_per_volume * 2,
        sampler=sampler,
        num_workers=2,
        shuffle_subjects=False,
        shuffle_patches=False,
        )
    

    training_loader_patches = torch.utils.data.DataLoader(
        patches_training_set, batch_size=training_batch_size)

    validation_loader_patches = torch.utils.data.DataLoader(
        patches_validation_set, batch_size=validation_batch_size)
    
    test_loader_patches = torch.utils.data.DataLoader(
        patches_test_set, batch_size=validation_batch_size)

    
    training_loader = torch.utils.data.DataLoader(
        training_set, batch_size=2)
    
    validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=1)
    
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1)
    

    if patches:
        return training_loader_patches, validation_loader_patches, test_loader_patches
    else:
        return training_loader, validation_loader, test_loader
    

from dpipe.torch.functional import dice_loss_with_logits
from dpipe.im.metrics import dice_score

model = Unet3d().to(config.device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters())

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='/home/alexey/Thesis/logs')


training_loader, validation_loader, test_loader = prepare_dataload(patches=False)
if config.continue_train:
    model.load_state_dict(torch.load('/home/alexey/Thesis/unet.pth'))

for epoch in range(config.num_epochs):
    
    epoch_loss = 0

    model.train()
    print('here')
    for batch in training_loader:
        X_batch, y_batch = batch['image'][tio.DATA], batch['mask'][tio.DATA]
        images, labels = X_batch.to(config.device), y_batch.to(config.device)
        
        pred = model(images)
        loss = criterion(pred, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.data.cpu().numpy() 

    
    writer.add_scalar('BCELoss', epoch_loss, epoch)
    torch.save(model.state_dict(), '/home/alexey/Thesis/unet.pth')

    model.eval()
    dices = []
    for batch in validation_loader:
        X_batch, y_batch = batch['image'][tio.DATA], batch['mask'][tio.DATA]
        images, labels = X_batch.to(config.device), y_batch.to(config.device)
        pred = model(images)
        
        dice = dice_loss_with_logits(pred, labels)
        dices.append(dice)
    writer.add_scalar('DICELoss', sum(dices)/len(dices), epoch)

model.eval()
dices = []
model.load_state_dict(torch.load('/home/alexey/Thesis/unet.pth'))

for batch in test_loader:
    X_batch, y_batch = batch['image'][tio.DATA], batch['mask'][tio.DATA]
    images, labels = X_batch.to(config.device), y_batch.to(config.device)
    pred = model(images).cpu().detach().numpy()
    pred = pred > 0.5

    labels = labels.cpu().detach().numpy().astype(bool)

    dice = dice_score(pred, labels)
    dices.append(dice)

print(f'Average DICE : {sum(dices)/len(dices)}')