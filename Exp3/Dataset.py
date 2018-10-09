import os
import numpy as np 
import imageio
from scipy import ndimage
from skimage.transform import resize

class Dataset():
    def __init__(self):
        self.dataClassList = []
        for _, b, _ in os.walk('caltech101'):
            if len(b) > 0:
                for dataClass in b:
                    if dataClass != 'BACKGROUND_Google':
                        self.dataClassList.append(dataClass)
        self.nClasses = len(self.dataClassList)
        self.x = np.empty((0, 240, 298))
        self.y = np.empty((0, self.nClasses))
        for classIdx, c in enumerate(self.dataClassList):
            for cPath, _, data in os.walk(os.path.join('caltech101', c)):
                for img in data:
                    image = imageio.imread(os.path.join(cPath, img), pilmode='L')
                    image = resize(image, (240, 298), mode='reflect', anti_aliasing=True)
                    image = np.expand_dims(image, 0)
                    target = np.zeros((1, self.nClasses))
                    np.put(target, classIdx, 1)
                    self.x = np.append(self.x, image, axis=0)
                    self.y = np.append(self.y, target, axis=0)

        print(self.x.shape)
        print(self.y.shape)


