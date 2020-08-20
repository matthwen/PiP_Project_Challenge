# -*- coding: utf-8 -*-

import numpy as np
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import PIL
import glob
import os
import utils
from PIL import Image
import cv2
import random
import torch


class ChallengeImagesReducedBatched(Dataset):
    def __init__(self, data_folder: str):
        files = sorted(glob.glob(os.path.join(data_folder, '**/*.jpg'), recursive=True))
        self.data = files

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_data = self.data[idx]
        # image_array, crop_array, target_array = utils.crop(np.array(Image.open(image_data)), (7, 11), (30, 50))
        # print(image_data)
        return image_data


class ChallengeImagesReduced(Dataset):
    def __init__(self, data_folder: str):
        files = sorted(glob.glob(os.path.join(data_folder, '**/*.jpg'), recursive=True))
        self.data = files

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_data = self.data[idx]

        # img = Image.open(image_data)
        img = cv2.imread(image_data, cv2.IMREAD_UNCHANGED)
        height = 90
        width = 90
        dim = (width, height)
        res = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
        crop_height = random.randrange(5, 21, 2)
        crop_width = random.randrange(5, 21, 2)
        image_array, crop_array, target_array = utils.crop(np.array(res),
                                                           (crop_height,
                                                            crop_width),
                                                           (random.randrange(20 + crop_height,
                                                                             height - 20 - crop_height),
                                                            random.randrange(20 + crop_width, width - 20 - crop_width)))
        # print(image_data)
        return image_array, crop_array, target_array, idx

class ChallengeImagesScoring(Dataset):
    def __init__(self, data: dict):
        self.data = data["images"]
        self.crop_sizes = data["crop_sizes"]
        self.crop_centers = data["crop_centers"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_data = self.data[idx]

        # img = Image.open(image_data)
        #img = cv2.imread(image_data, cv2.IMREAD_UNCHANGED)
        # height = 90
        # width = 90
        # dim = (width, height)
        # res = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
        # crop_height = random.randrange(5, 21, 2)
        # crop_width = random.randrange(5, 21, 2)
        img = Image.fromarray(image_data)
        image = np.array(img)
        mean = image.mean()
        std = image.std()
        image = (image - mean) / std

        image_array, crop_array, target_array = utils.crop(np.array(image),
                                                           self.crop_sizes[idx],self.crop_centers[idx])
        # print(image_data)
        return image_array, crop_array, target_array, mean, std,self.crop_sizes[idx],self.crop_centers[idx]

def collate_fn(image_data_list: list):
    # targets = [sample[2] for sample in batch_as_list]
    # max_h = np.max([target.shape[0] for target in targets])
    # max_w = np.max([target.shape[1] for target in targets])
    # stacked_targets = torch.zeors(size=(len(targets), max_h, max_w))
    # for i, target in enumerate(targets):
    #    stacked_targets[i,]
    # return torch.stack(batch_as_list[0]), torch.stack(batch_as_list[1])

    height = random.randrange(70, 100, 1)
    width = random.randrange(70, 100, 1)
    dim = (width, height)
    crop_height = random.randrange(5, 21, 2)
    crop_width = random.randrange(5, 21, 2)
    crop_center_x = random.randrange(20 + (crop_height // 2), height - 20 - (crop_height // 2))
    crop_center_y = random.randrange(20 + (crop_width // 2), width - 20 - (crop_width // 2))

    image_list = []
    crop_list = []
    target_list = []
    means = []
    stds = []
    # images = torch.tensor(batch_size,height,width)
    # crops = torch.tensor(batch_size,height,width)
    # targets = torch.tensor(batch_size,height,width)

    for image_data in image_data_list:
        img = cv2.imread(image_data, cv2.IMREAD_UNCHANGED)
        res = cv2.resize(img, dim, interpolation=cv2.INTER_LANCZOS4)
        # img = Image.open(image_data)
        # transorm = torchvision.transforms.Resize(dim,Image.LANCZOS)
        image = np.array(res)
        mean = image.mean()
        std = image.std()
        image = (image - mean) / std
        means.append(mean)
        stds.append(std)
        image_array, crop_array, target_array = utils.crop(np.array(image),
                                                           (crop_height,
                                                            crop_width),
                                                           (crop_center_x, crop_center_y))
        image_list.append(torch.tensor(image_array))
        crop_list.append(torch.tensor(crop_array))
        target_list.append(torch.tensor(target_array))
    return torch.stack(image_list), torch.stack(crop_list), torch.stack(target_list), means, stds
