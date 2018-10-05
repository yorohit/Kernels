import numpy as np 
from sklearn import preprocessing
import itertools
import cPickle
from sklearn.svm import SVC
import performance_metrics
from sklearn.model_selection import train_test_split
import matplotlib
from pandas import DataFrame
from matplotlib import pyplot
from PIL import Image

X=[]
Y=[]

def SVM(x_train, y_train, x_test, y_test):
	clf = SVC(kernel = 'linear')
	print "Linear kernel"
	clf.fit(x_train, y_train)
	y_pred = clf.predict(x_test)
	print performance_metrics.performance(y_test, y_pred)

	print "rbf kernel"
	clf = SVC(kernel = 'rbf',C=1,gamma=0.1)
	clf.fit(x_train, y_train)
	y_pred = clf.predict(x_test)
	print performance_metrics.performance(y_test, y_pred)
	print "polynomial kernel"
	clf = SVC(kernel = 'poly',degree=2,C=10)
	clf.fit(x_train, y_train)
	y_pred = clf.predict(x_test)
	print performance_metrics.performance(y_test, y_pred)


with open("/home/rohit/kernel_project/dataset/swiss_roll.txt", "r") as filestream:
	for line in filestream:
		currentline = line.split("  ")
		#print currentline
		#break
		x_t=[float(currentline[1]),float(currentline[2]),float(currentline[3])]
		X.append(x_t)
		#Y.append(int(currentline[2]))

with open("/home/rohit/kernel_project/dataset/swiss_roll_labels.txt", "r") as filestream:
	for line in filestream:
		currentline = line
		#x_t=[float(currentline[0]),float(currentline[1])]
		#X.append(x_t)
		Y.append(int(float(currentline)))


X=np.array(X)
Y=np.array(Y)

img = Image.fromarray(X, 'RGB')
img.save('my.png')
img.show()

# df = DataFrame(dict(x=X[:,0], y=X[:,1], z=X[:,2],label=Y))
# colors = {0:'red', 1:'blue', 2:'green', 3:'black', 4:'yellow', 5:'pink'}
# fig, ax = pyplot.subplots()
# grouped = df.groupby('label')
# for key, group in grouped:
# 	group.plot(ax=ax, kind='scatter', x='x', y='y',z='z', label=key, color=colors[key])
# pyplot.show()
#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
#SVM(x_train, y_train, x_test, y_test)