from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np 
from sklearn import preprocessing
import itertools


arr = np.random.randint(0,1, 100*100).astype(np.uint8)
arr.resize((100,100))
print arr
img = Image.fromarray(arr*255, mode='L')
#img.save(name)
img.show()