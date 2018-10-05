import numpy as np 
from sklearn import preprocessing
import itertools
import cPickle
from sklearn.svm import SVC
import performance_metrics
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt

X=[]
Y=[]

def SVM(x_train, y_train, x_test, y_test):
	
	# clf = SVC(kernel = 'linear')
	# print "Linear kernel"
	# clf.fit(x_train, y_train)
	# y_pred = clf.predict(x_test)
	# print performance_metrics.performance(y_test, y_pred)

	print "rbf kernel"
	clf = SVC(kernel = 'rbf',C=1,gamma=0.1)
	clf.fit(x_train, y_train)
	y_pred = clf.predict(x_test)
	print performance_metrics.performance(y_test, y_pred)
	
	#print y_test[1:50],y_pred[1:50]
	# print "polynomial kernel"
	# clf = SVC(kernel = 'poly',degree=3,C=10)
	# clf.fit(x_train, y_train)
	# y_pred = clf.predict(x_test)
	# print performance_metrics.performance(y_test, y_pred)
	return clf

#load_data
with open("/home/rohit/kernel_project/dataset/DoubleMoon1.txt", "r") as filestream:
	for line in filestream:
		currentline = line.split(",")
		x_t=[float(currentline[0]),float(currentline[1])]
		X.append(x_t)
		Y.append(int(currentline[2]))

#print X[0],Y[0]
X=np.array(X)

Y=np.array(Y)
colors = ['red','green','blue','purple']
fig = plt.figure(figsize=(8,8))
plt.scatter(X[:,0], X[:,1], c=Y, cmap=matplotlib.colors.ListedColormap(colors))

cb = plt.colorbar()
loc = np.arange(0,max(Y),max(Y)/float(len(colors)))
cb.set_ticks(loc)
cb.set_ticklabels(colors)
# plt.show()
#print X[1],Y[1]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
h = .02 
clf=SVM(x_train, y_train, x_test, y_test)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
plt.show()

