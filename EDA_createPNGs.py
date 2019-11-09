
# coding: utf-8

# In[1]:


import pydicom
import os
import matplotlib.pyplot as plt
import numpy as np

import csv
import cv2
import PIL
import tqdm


# In[2]:


labelDict = {}

with open('locationData.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader)
    for row in csv_reader:
        identifier = str(row[0]).zfill(4)
        tubePresentBool = row[1]=='Y'
        if tubePresentBool:
            xCoord = float(row[2])
            yCoord = float(row[3])
        else:
            xCoord = 0.0
            yCoord = 0.0
        labelDict[identifier] = [tubePresentBool, xCoord, yCoord]


# In[ ]:


IDlist = list(labelDict.keys())


# In[ ]:


missingIDs = []

for eachID in tqdm.tqdm(IDlist):  
    try:
        ds = pydicom.read_file('DICOMS/'+eachID + '.dcm')
        pix = ds.pixel_array
        img = PIL.Image.fromarray(np.uint8((pix/pix.max())*255))
        img.save('PNG_all/'+ eachID + '.png')
    except:
        missingIDs.append(eachID)


# In[ ]:


print(missingIDs)


# In[ ]:


for eachID in tqdm.tqdm(IDlist):
    if labelDict[eachID][0]:
        try:
            ds = pydicom.read_file('DICOMS/'+eachID + '.dcm')
            pix = ds.pixel_array

            origX = labelDict[eachID][1]
            origY = labelDict[eachID][2]

            cv2.circle(pix,(int(origX), int(origY)), 100, (0,255,0), 5)
            img = PIL.Image.fromarray(np.uint8((pix/pix.max())*255))
            img.save('PNG_tubesAnnotated/'+ eachID + '.png')

