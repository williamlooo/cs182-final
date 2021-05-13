import cv2
import numpy as np
from PIL import Image

class RandomGaussianBlur(object):
    """
    Implements random gaussian blur with kernel. a kernel of (0,0) is an identity blur
    """
    def __init__(self, random_blur_gaussian=True, kernel=(3, 3), stdX=0):
        self.random_blur_gaussian:bool = random_blur_gaussian
        self.kernel:tuple = kernel
        self.stdX:float = 0
    def __call__(self, img):
        img = np.array(img)
        if self.random_blur_gaussian:
            #the number 3 is empirically determined to be an appropriate amount of blur
            random_filter_dim = np.random.randint(0, 3) 
            #to avoid shifting, kernel dimensions have to be odd, so if 2 is sampled, kernel is set to (3,3): 
            # https://stackoverflow.com/questions/52348769/what-if-the-filter-window-size-is-an-even-number-in-gaussian-filtering
            kernel = (random_filter_dim,random_filter_dim) if random_filter_dim == 0 or random_filter_dim % 2 == 1 else (random_filter_dim+1,random_filter_dim+1)
        else:
            kernel = self.kernel

        if kernel == (0,0):
            #do nothing
            blur_img = img
        else:
            blur_img = cv2.GaussianBlur(img.copy(), kernel, self.stdX)
        return Image.fromarray(blur_img)
    def __repr__(self):
        return self.__class__.__name__+'()'


import numpy as np
import PIL

#https://www.simonwenkel.com/2019/11/13/Salt-and-Pepper-Noise-for-data-augmentation.html
class RandomSaltAndPepperNoise(object):
    r""" Implements 'Salt-and-Pepper' noise
    Adding grain (salt and pepper) noise
    (https://en.wikipedia.org/wiki/Salt-and-pepper_noise)

    assumption: high values = white, low values = black
    
    Inputs:
            - rate (float): should we apply or not?
            - threshold (float):
            - lowerValue (int): value for "pepper"
            - upperValue (int): value for "salt"
            - noiseType (str): {"SnP", "RGB"}
    Output:
            - image ({np.ndarray, PIL.Image}): image with 
                                               noise added
    """
    def __init__(self,
                rate:float=0.1,
                 threshold:float = 0.005,
                 lowerValue:int = 5,
                 upperValue:int = 250,
                 noiseType:str = "RGB"):
        self.rate = rate
        self.threshold = threshold
        self.lowerValue = lowerValue # 255 would be too high
        self.upperValue = upperValue # 0 would be too low
        self.noiseType = noiseType
        if self.noiseType != "RGB":
            print("warning, only RGB noise supported.")
        super(RandomSaltAndPepperNoise).__init__()

    def __call__(self, img):
        if np.random.rand() < self.rate:
            img = np.array(img)
            if type(img) != np.ndarray:
                raise TypeError("Image is not of type 'np.ndarray'!")
            random_matrix = np.random.random(img.shape)      
            img[random_matrix>=(1-self.threshold)] = self.upperValue
            img[random_matrix<=self.threshold] = self.lowerValue
            return PIL.Image.fromarray(img)

        return img
