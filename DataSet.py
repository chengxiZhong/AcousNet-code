import os
import numpy as np
import torch
import torch.utils.data as Data
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
from PIL import Image


root_path = "Concate"
class MyDataset(Dataset):

    def __init__(self, Y_path, X_path, data_type, Y_transforms = None, X_transforms = None):
        
        Y_fh = open(Y_path, 'r')
        X_fh = open(X_path, 'r')
        data_Y_X = []
        for data_Y, data_X in zip(Y_fh, X_fh):
            data_Y = data_Y.rstrip()
            data_X = data_X.rstrip()
            data_Y_X.append((data_Y, data_X))

        self.data_Y_X = data_Y_X
        self.data_type = data_type
        self.X_transforms = X_transforms
        self.Y_transforms = Y_transforms

    def __getitem__(self, index):

        Y_path, X_path = self.data_Y_X[index]

        with open(os.path.join(root_path, f'{self.data_type}_Normalized_Y', Y_path), 'rb') as Y:
            data_Y = np.load(Y)
            data_Y = data_Y['normalized_pap'].astype(np.float32)
            data_Y = torch.from_numpy(data_Y)
        data_Y = data_Y[np.newaxis, 3:]

        with open(os.path.join(root_path, f'{self.data_type}_Normalized_X', X_path), 'rb') as X:
            data_X = np.load(X)
            data_X = data_X['normalize_x_pahse'].astype(np.float32)
            data_X = torch.from_numpy(data_X)
        data_X = data_X.reshape(2500)

        return data_Y, data_X

    def __len__(self):
        return len(self.data_Y_X)


BATCH_SIZE = 32
Validation_Split = .2
SHUFFLE = True
Random_Seed = 42



nw = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])   # num_workers = 8
print('Using {} dataloader workers every process'.format(nw))

transform_Y = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor()
    ]
)
transform_X = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor()
    ]
)


# train dataset
train_Y_path = "Concate/Y_new_filenames_4800.txt"
train_X_path = "Concate/X_new_filenames_4800.txt"

train_dataset = MyDataset(train_Y_path, train_X_path, "New", transform_Y, transform_X)

# test dataset
test_Y_path = "Concate/Y_new_filenames_1200.txt"
test_X_path = "Concate/X_new_filenames_1200.txt"
test_dataset = MyDataset(test_Y_path, test_X_path, "New", transform_Y, transform_X)


# Encapsulate training input and label into Data.TensorDataset() class object
# train_dataset = Data.TensorDataset(train_Y, train_X)
# Put train_dataset into DataLoader
train_loader = Data.DataLoader(
    dataset=train_dataset, # Data
    batch_size=BATCH_SIZE, # Batch size
    shuffle=SHUFFLE, # shuffle dataset
    num_workers=nw, # Apply multiprocess to read data
    pin_memory=False,
)

# Encapsulate testing input and label into Data.TensorDataset() class object
# test_dataset = Data.TensorDataset(test_Y, test_X)
# Put test_dataset into DataLoader
valid_loader = Data.DataLoader(
    dataset=test_dataset, # Data
    batch_size=BATCH_SIZE, # Batch size
    shuffle=False, # shuffle dataset
    num_workers=nw, # Apply multiprocess to read data
    pin_memory=False,
    # drop_last=True,
)


# Both dataset
both_Y_path = "Concate/Y_both_filenames_2w.txt"
both_X_path = "Concate/X_both_filenames_2w.txt"
both_dataset = MyDataset(both_Y_path, both_X_path, "Both", transform_Y, transform_X)
n_train = len(both_dataset) * 0.8
n_test = len(both_dataset) * 0.2
print("using {} Y for training, {} Y for validation.".format(n_train, n_test))

# Create data indices for training and validation splits:
Dataset_Size = len(both_dataset)
indices = list(range(Dataset_Size))
split = int(np.floor(Validation_Split * Dataset_Size)) # 0.2 * 2w = 4k
if SHUFFLE:
    np.random.seed(Random_Seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]  # train_indices = [4k:], val_indices = [0:4k]

# Creating PT data samplers and loaders:
train_sampler = Data.SubsetRandomSampler(train_indices)
valid_sampler = Data.SubsetRandomSampler(val_indices)

train_loader = Data.DataLoader(
    dataset = both_dataset,
    batch_size = BATCH_SIZE,
    sampler = train_sampler,
)

valid_loader = Data.DataLoader(
    dataset = both_dataset,
    batch_size = BATCH_SIZE,
    sampler = valid_sampler,
)
