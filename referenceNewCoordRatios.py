
# coding: utf-8

# In[1]:


import os
import pydicom
import csv
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import matplotlib
import pickle
from PIL import Image
import cv2


xRatio = {}
yRatio = {}

with open('newCoordinateRatio.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader)
    for row in csv_reader:
        if (row[1]):
            ptID = row[0].zfill(4)
            xRatio[ptID] = float(row[1])
            yRatio[ptID] = float(row[2])


# In[22]:


sampleID = listOfTubeID[20]

img = Image.open('PNG256/'+sampleID+'.png')
img = np.asarray(img)
cv2.circle(img,(int(xRatio[sampleID]*256), int(yRatio[sampleID]*256)), 15, (0,255,0), 1)
plt.figure(figsize=(15,15))
plt.imshow(img, cmap='gray')

