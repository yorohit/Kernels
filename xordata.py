import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC

def features3d(X):
	z=np.abs(X[:,0]+X[:,1])
	res=[]
	for i,x in enumerate(X):
		res.append([x[0],x[1],z[i]])
	return res


def mykernel(X,Y):
	x_n=features3d(X)
	x_n=np.array(x_n)
	y_n=features3d(Y)
	y_n=np.array(x_n)
	return np.dot(x_n, y_n.T)


def SVM(x,y):

	#gram = np.dot(xx, xx.T)
	#print gram


	clf = SVC(kernel=mykernel)
	print "Manual kernel"
	clf.fit(x, y)
	

	# clf = SVC(kernel = 'linear')
	# print "Linear kernel"
	# clf.fit(x, y)
	#y_pred = clf.predict(x_test)
	#print performance_metrics.performance(y_test, y_pred)

	# print "rbf kernel"
	# clf = SVC(kernel = 'rbf',C=10,gamma=10)
	# clf.fit(x_train, y_train)
	# y_pred = clf.predict(x_test)
	# print performance_metrics.performance(y_test, y_pred)
	#print y_test[1:50],y_pred[1:50]
	

	# print "polynomial kernel"
	# clf = SVC(kernel = 'poly',degree=3,C=10)
	# clf.fit(x_train, y_train)
	# y_pred = clf.predict(x_test)
	# print performance_metrics.performance(y_test, y_pred)
	
	return clf



#rng = np.random.RandomState(0)
X = np.array([[1,1],[-1,-1],[1,-1],[-1,1]])

Y = np.array([1,1,0,0])
#print X.shape
# x_n=features3d(X)
# x_n=np.array(x_n)
#print x_n.shape

svc=SVM(X ,Y)

colors = ['red','green','blue','purple']
fig = plt.figure(figsize=(8,8))
plt.scatter(X[:,0], X[:,1], c=Y, cmap=matplotlib.colors.ListedColormap(colors))

cb = plt.colorbar()
loc = np.arange(0,max(Y),max(Y)/float(len(colors)))
cb.set_ticks(loc)
cb.set_ticklabels(colors)

h = .02 
#clf=SVM(x_train, y_train, x_test, y_test)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
plt.show()







# z = lambda x,y: (-svc.intercept_[0]-svc.coef_[0][0]*x-svc.coef_[0][1]) / svc.coef_[0][2]

# tmp = np.linspace(-2,2,51)
# x,y = np.meshgrid(tmp,tmp)

# # Plot stuff.
# fig = plt.figure()
# ax  = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x, y, z(x,y))
# ax.plot3D(X[Y==0,0], X[Y==0,1], X[Y==0,2],'ob')
# ax.plot3D(X[Y==1,0], X[Y==1,1], X[Y==1,2],'sr')
# plt.show()


