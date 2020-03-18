from torch.utils.data.dataloader import DataLoader

import albumentations as albu
from albumentations import pytorch as AT

from Dataset import MIDataset

train_dataset = MIDataset(datatype='train', transforms=albu.Compose([])) # , preprocessing=get_preprocessing(preprocessing_fn))
valid_dataset = MIDataset(folder="dataset_crf/valid", datatype='valid', transforms=albu.Compose([]))  # , preprocessing=get_preprocessing(preprocessing_fn))

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

for i, _ in enumerate(train_loader):
    print(i)
for i, _ in enumerate(valid_loader):
    print(i)