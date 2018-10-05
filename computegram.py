import numpy as np 
from sklearn import preprocessing
import itertools
import cPickle
import seaborn as sns
from sklearn.svm import SVC
import performance_metrics
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import distance
from PIL import Image

X=[]
Y=[]
#sns.set()

def create_img(gram,name):
	# img = Image.fromarray(gram, mode='L')
	# img.save(name)
	# img.show()
    plt.imshow(gram)
    plt.colorbar()
    plt.show()
    # ax = sns.heatmap(gram)
    # figure = ax.get_figure()    
    # figure.savefig(name)


def linear(X):
	gram = np.dot(X, X.T)
	return gram

def polynomial(x,z,degree,c):
    gram = np.dot(x,z)+c
    res=1
    for i in range(0,degree):
        res=res*gram

    return res


def Gaussian(x,z,sigma):
    return np.exp((-(np.linalg.norm(x-z)**2))/(2*sigma**2))

def euclid(x,z):
    return distance.euclidean(x,z)

def GaussianMatrix(X,sigma):
    row,col=X.shape
    GassMatrix=np.zeros(shape=(row,row))
    X=np.asarray(X)
    i=0
    for v_i in X:
        j=0
        for v_j in X:
            GassMatrix[i,j]=Gaussian(v_i.T,v_j.T,sigma)
            #GassMatrix[i,j]=euclid(v_i.T,v_j.T)
            #GassMatrix[i,j]=polynomial(v_i.T,v_j.T,3,10)
            j+=1
        i+=1
    return GassMatrix


# with open("/home/rohit/kernel_project/dataset/TwoSpirals.txt", "r") as filestream:
#     for line in filestream:
#         currentline = line.split(" ")
#         #print line
#         x_t=[float(currentline[0]),float(currentline[1])]
#         #if(int(currentline[2])==0):
#         X.append(x_t)
#         Y.append(int(currentline[2]))

with open("/home/rohit/kernel_project/dataset/swiss_roll.txt", "r") as filestream:
    for line in filestream:
        currentline = line.split("  ")
        #print currentline
        #break
        x_t=[float(currentline[1]),float(currentline[2]),float(currentline[3])]
        X.append(x_t)


X=np.array(X)
#Y=np.array(Y)
#gram=linear(X)
gram=GaussianMatrix(X,0.001)
#print gram
#gram=((gram))#.astype(np.uint8)
#print gram
create_img(gram,'Corners_rgb_euclid_2.png')