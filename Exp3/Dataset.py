import os
import numpy as np 
from scipy import ndimage
from keras.preprocessing.image import load_img

class Dataset():
    def __init__(self):
        self.folderName = 'caltech101_test'
        self.bgFolder = 'BACKGROUND_Google'
        self.imgSize = [240, 300]

        self.dataClassList = os.listdir(self.folderName)
        self.dataClassList.remove(self.bgFolder)
        
        self.nClasses = len(self.dataClassList)
        self.data = {}
        self.target = {}
        
        for classIdx, c in enumerate(self.dataClassList):
            self.data[c] = np.empty([0] + self.imgSize)
            self.target[c] = np.zeros((1, self.nClasses))
            np.put(self.target[c], classIdx, 1)
            

            cPath = os.path.join(self.folderName, c)
            for img in os.listdir(cPath):
                imgPath = os.path.join(cPath, img)
                image = load_img(imgPath, color_mode='grayscale', target_size=self.imgSize)
                image = np.expand_dims(image, 0)
                self.x = np.append(self.x, image, axis=0)
                self.y = np.append(self.y, target, axis=0)

    def getData(self, train=0.7, validation=0.2, test=0.1):
        if train+validation+test != 1:
            print('error: train+validation+test must add up to 1.0')
            return None, None, None, None
