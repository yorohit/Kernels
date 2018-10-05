import numpy as np 
from sklearn import preprocessing
from keras.datasets import cifar10 as c
from sklearn.preprocessing import label_binarize
import itertools
import cPickle
from sklearn.svm import SVC
import performance_metrics
dataset = "C10K"
split = "70-30"
#import read_data
def c1(split):

	num_class = 10
	
	num_train_per_class = int(int(split[0:2]) * 0.01 * 100)
	num_test_per_class = int(int(split[3:5]) * 0.01 * 100)

	train_file = "/home/rohit/pro/Features/C1K/" + split + "/c1_cnn_train.npy"
	test_file = "/home/rohit/pro/Features/C1K/" + split + "/c1_cnn_test.npy"

	print "OK"

	x_train = np.load(open(train_file))
	x_test = np.load(open(test_file))

	labels = []
	for i in xrange(num_class):
		labels.append([i] * num_train_per_class)
	y_train = np.array(list(itertools.chain.from_iterable(labels)))
	#y_train = preprocessing.label_binarize(y_train, classes = list(np.arange(num_class)))

	labels = []
	for i in xrange(num_class):
		labels.append([i] * num_test_per_class)
	y_test = np.array(list(itertools.chain.from_iterable(labels)))
	#y_test = preprocessing.label_binarize(y_test, classes = list(np.arange(num_class)))

	return x_train, y_train, x_test, y_test

def c10(split):

	num_class = 100
	num_train_per_class = int(int(split[0:2]) * 0.01 * 100)
	num_test_per_class = int(int(split[3:5]) * 0.01 * 100)

	train_file = "/home/rohit/pro/Features/C10K/" + split + "/c10_cnn_train.npy"
	test_file = "/home/rohit/pro/Features/C10K/" + split + "/c10_cnn_test.npy"

	x_train = np.load(open(train_file))
	x_test = np.load(open(test_file))

	labels = []
	for i in xrange(num_class):
		labels.append([i] * num_train_per_class)
	y_train = np.array(list(itertools.chain.from_iterable(labels)))
	#y_train = preprocessing.label_binarize(y_train, classes = list(np.arange(num_class)))

	labels = []
	for i in xrange(num_class):
		labels.append([i] * num_test_per_class)
	y_test = np.array(list(itertools.chain.from_iterable(labels)))
	#y_test = preprocessing.label_binarize(y_test, classes = list(np.arange(num_class)))

	return x_train, y_train, x_test, y_test

def ghim(split):

	num_class = 20
	
	num_train_per_class = int(int(split[0:2]) * 0.01 * 500)
	num_test_per_class = int(int(split[3:5]) * 0.01 * 500)

	train_file = "/home/rohit/pro/Features/GHIM-10K/" + split + "/ghim_cnn_train.npy"
	test_file = "/home/rohit/pro/Features/GHIM-10K/" + split + "/ghim_cnn_test.npy"

	x_train = np.load(open(train_file))
	x_test = np.load(open(test_file))

	labels = []
	for i in xrange(num_class):
		labels.append([i] * num_train_per_class)
	y_train = np.array(list(itertools.chain.from_iterable(labels)))
	#y_train = preprocessing.label_binarize(y_train, classes = list(np.arange(num_class)))

	labels = []
	for i in xrange(num_class):
		labels.append([i] * num_test_per_class)
	y_test = np.array(list(itertools.chain.from_iterable(labels)))
	#y_test = preprocessing.label_binarize(y_test, classes = list(np.arange(num_class)))

	return x_train, y_train, x_test, y_test


def SVM(x_train, y_train, x_test, y_test):
	#clf = SVC(kernel = 'rbf',C=50,gamma=0.0005)
	#clf = SVC(kernel = 'poly',degree=3,C=50)
	clf = SVC(kernel = 'linear')
	print "Linear kernel"
	clf.fit(x_train, y_train)
	y_pred = clf.predict(x_test)

	print performance_metrics.performance(y_test, y_pred)

	print "rbf kernel"
	clf = SVC(kernel = 'rbf',C=50,gamma=0.0005)
	clf.fit(x_train, y_train)
	y_pred = clf.predict(x_test)
	
	print performance_metrics.performance(y_test, y_pred)

	print "polynomial kernel"
	clf = SVC(kernel = 'poly',degree=2,C=10)
	clf.fit(x_train, y_train)
	y_pred = clf.predict(x_test)
	
	print performance_metrics.performance(y_test, y_pred)

def cifar10(p):

	num_class = 10
	# num_train_per_class = 5000
	# num_test_per_class = 200

	# train_file = "/home/rohit/pro/Features/CIFAR10/cifar_cnn_train.npy"
	# test_file = "/home/rohit/pro/Features/CIFAR10/cifar_cnn_test.npy"

	# x_train = np.load(open(train_file))
	# x_test = np.load(open(test_file))

	# y_train = np.load(open("/home/rohit/pro/Features/CIFAR10/cifar_cnn_train_label.npy"))
	# y_test = np.load(open("/home/rohit/pro/Features/CIFAR10/cifar_cnn_test_label.npy"))

	(_, y_train), (_, y_test) = c.load_data()
	#y_train = label_binarize(y_train, classes = list(np.arange(10)))
	#y_test = label_binarize(y_test, classes = list(np.arange(10)))

	path = '/home/rohit/pro/Features/CIFAR10/'

	for i in range(1, 11):
		if i == 1:
			x_train = np.load(path + 'train/bot_train_' + str(i))
		else:
			temp = np.load(path + 'train/bot_train_' + str(i))
			x_train = np.append(x_train, temp, axis = 0)

	for i in range(1, 3):
		if i == 1:
			x_test = np.load(path + 'val/bot_test_' + str(i))
		else:
			temp = np.load(path + 'val/bot_test_' + str(i))
			x_test = np.append(x_test, temp, axis = 0)			

	return x_train, y_train, x_test, y_test


#x_train, y_train, x_test, y_test = cifar10(split)

with open("/home/rohit/pro/Features/CIFAR10/PCA/50/"+ "x_test_red", "rb") as f:
	x_test = cPickle.load(f)

with open("/home/rohit/pro/Features/CIFAR10/PCA/50/"+ "x_train_red", "rb") as f:
	x_train = cPickle.load(f)
#print x_train.shape, y_train.shape
x_train, y_train, x_test, y_test = cifar10(split)
x_train= np.reshape(x_train,(x_train.shape[0],-1))
x_test= np.reshape(x_test,(x_test.shape[0],-1))
SVM(x_train, y_train, x_test, y_test)


