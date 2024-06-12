import torch

import cv2
import PIL
data = 1000

import csv
from PIL import Image



conv1 = torch.nn.Conv2d(1,16)
active1 = torch.nn.ReLU()
conv2 = torch.nn.Conv2d(16,32)

linear = torch.nn.Linear(14,32)
x = [[0][0]]
x = conv1(x)
x = active1(x)
x = conv2(x)


