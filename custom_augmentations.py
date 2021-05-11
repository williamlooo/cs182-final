import cv2
import numpy as np
from PIL import Image, ImageEnhance

class RandomGaussianBlur(object):
    def __init__(self, random_blur_gaussian=True, kernel=(5, 5), stdX=0):
        self.random_blur_gaussian = random_blur_gaussian
        self.kernel = kernel
        self.stdX = 0
    def __call__(self, img):
        img = np.array(img)
        if self.random_blur_gaussian:
            #the number 3 is empirically determined to be an appropriate amount of blur
            random_filter_dim = np.random.randint(0, 3) 
            #to avoid shifting, kernel dimensions have to be odd, so if 2 is sampled, kernel is set to (3,3): 
            # https://stackoverflow.com/questions/52348769/what-if-the-filter-window-size-is-an-even-number-in-gaussian-filtering
            kernel = (random_filter_dim,random_filter_dim) if random_filter_dim % 2 == 1 else (random_filter_dim+1,random_filter_dim+1)
        else:
            kernel = self.kernel
        blur_img = cv2.GaussianBlur(img.copy(), kernel, self.stdX)
        return Image.fromarray(blur_img)
    def __repr__(self):
        return self.__class__.__name__+'()'