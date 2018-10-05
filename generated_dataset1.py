from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets.samples_generator import make_moons
from sklearn.datasets.samples_generator import make_circles
from matplotlib import pyplot
from pandas import DataFrame
import numpy as np 
from sklearn import preprocessing
import itertools
import cPickle
from sklearn.svm import SVC
import performance_metrics
from sklearn.model_selection import train_test_split
# generate 2d classification dataset

#X, Y = make_circles(n_samples=1000, noise=0.1, random_state=41)
X, Y = make_moons(n_samples=1000, noise=0.6,random_state=42)
#X, Y = make_blobs(n_samples=1000, centers=6, n_features=2,random_state=30)

X=np.array(X)
Y=np.array(Y)

#print Y
def SVM(x_train, y_train, x_test, y_test):
	# clf = SVC(kernel = 'linear')
	# print "Linear kernel"
	# clf.fit(x_train, y_train)
	# y_pred = clf.predict(x_test)
	# print performance_metrics.performance(y_test, y_pred)

	print "rbf kernel"
	clf = SVC(kernel = 'rbf',C=10,gamma=10)
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




#scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1],label=Y))
colors = {0:'red', 1:'blue', 2:'green', 3:'black', 4:'yellow', 5:'pink'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
#pyplot.show()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
#clf=SVM(x_train, y_train, x_test, y_test)
#h = .02 
clf=SVM(x_train, y_train, x_test, y_test)
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                      np.arange(y_min, y_max, h))

# Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# pyplot.contour(xx, yy, Z, cmap=pyplot.cm.Paired)
# pyplot.show()

