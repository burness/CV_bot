'''
Coding Just for Fun
Created by burness on 16/9/7.
'''
from PIL import Image
import numpy as np
img = Image.open("4be532e1cd814731b998ff8f9e97707a.jpg")
img.load()
img = np.asarray(img)
print img.shape