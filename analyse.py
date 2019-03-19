#!/usr/bin/env python

import numpy as np
import pandas as pd
import argparse
import csv
import matplotlib.pyplot as plt
import string, math, os, random
from mpl_toolkits.mplot3d import Axes3D
from plotly import tools
import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import sklearn
from sklearn import datasets, model_selection, metrics, neighbors, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from matplotlib.colors import ListedColormap


def load_data(input_name):
	'''Loading data from sklearn'''
	names = [name for name in dir(sklearn.datasets) if name.startswith("load")]
	assert "load_{0}".format(input_name) in names, 'Invalid dataset name: ' + input_name + '\nPossible names: \nboston \nwine \niris \ndiabetes \nbreast_cancer'
	
	for i in names:
		if input_name == 'boston':
			from sklearn.datasets import load_boston
			dataset = load_boston()
			classification_flag = False			# For future grouping purposes
		elif input_name == 'wine':
			from sklearn.datasets import load_wine
			dataset = load_wine()		
			classification_flag = True	
		elif input_name == 'iris':
			from sklearn.datasets import load_iris
			dataset = load_iris()
			classification_flag = True
		elif input_name == 'diabetes':
			from sklearn.datasets import load_diabetes
			dataset = load_diabetes()
			classification_flag = False		
		elif input_name == 'breast_cancer':
			from sklearn.datasets import load_breast_cancer
			dataset = load_breast_cancer()
			classification_flag = True	
	print('Successfully loaded dataset ', input_name)
	return(dataset, classification_flag)

	
def parser_assign():
	'''Setting up parser for the file name and header file name '''
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset_name")   # name of the file specified in Dockerfile
	args = parser.parse_args()
	d_name = args.dataset_name
	return d_name


def read_data(df, feature_n, tar):
	'''Copying data from dataset to Data Frame'''
	data = pd.DataFrame(data = df)
	data.columns = feature_n			# assigning feature names to the names of the columns
	try:
		data['target'] = pd.Categorical(pd.Series(tar).map(lambda x: dataset.target_names[x]))
	except:
		data['target'] = tar
	
	if dataset_name == 'breast_cancer':		# if this is breast cancer dataset we choose only mean values for visalisation (10 out of 30 features)
		data1 = data.iloc[:,:10]
		data1['target'] = data['target']
	else: data1 = data

	grouped = dict()						# Defining a dictionary of grouped elements for future usage
	if classification_flag == True:
		d = []
		l = []
		for i, name in enumerate(dataset.target_names):
			d.append(data1.loc[data1['target']==name])
			l.append(name)
		grouped["data"] = d
		grouped["labels"] = l
	return data1, grouped


def all_functions(c_flag, df, gr):			#Closure that takes classification_flag, dataframe and grouped dictionary as an input
	def find_mean_std():
		'''Calculating mean and std for each of 30 features'''
		ave_feature = np.mean(df) 		
		std_feature = np.std(df) 

		print('\n ave of each measurment:\n', ave_feature)
		print('\n std of each measurment:\n', std_feature)

	
	def plot_histograms():
		'''Histogram all in one figure'''
		folder = "hist_{0}".format(dataset_name)
		if not os.path.exists(folder):
			os.makedirs(folder)
		columns = df.columns
		l = len(columns)
		n_cols = math.ceil(math.sqrt(l))		#Calculating scaling for any number of features
		n_rows = math.ceil(l / n_cols)
		
		fig=plt.figure(figsize=(11, 6), dpi=100)
		for i, col_name in enumerate(columns):
			if (classification_flag == False):
				ax=fig.add_subplot(n_rows,n_cols,i+1)
				df[col_name].hist(bins=10,ax=ax)
				ax.set_title(col_name)
			elif col_name != 'target':
				ax=fig.add_subplot(n_rows,n_cols,i+1)
				df[col_name].hist(bins=10,ax=ax)
				ax.set_title(col_name)
		fig.tight_layout() 
		plt.savefig("./{0}/all_hist.png".format(folder), bbox_inches='tight')
		plt.show()


	def plot_histograms_grouped():
		"""Histogram: all features in one figure grouped by one element"""
		folder = "hist_{0}".format(dataset_name)
		if not os.path.exists(folder):
			os.makedirs(folder)
		columns = df.columns
		l = len(df.columns)-1
		n_cols = math.ceil(math.sqrt(l))		# Calculating scaling for any number of features
		n_rows = math.ceil(l / n_cols)
		
		fig=plt.figure(figsize=(11, 6), dpi=100)
		
		idx = 0
		for i, col_name in enumerate(df.columns):		# Going through all the features
			idx = idx+1
			if col_name != 'target':				# Avoiding a histogram of the grouping element
				ax=fig.add_subplot(n_rows,n_cols,idx)
				ax.set_title(col_name)
				group = df.pivot(columns='target', values=col_name)
				for j, gr_feature_name in enumerate(group.columns):			# Going through the values of grouping feature (here malignant and benign)
					group[gr_feature_name].hist(alpha=0.5, label=gr_feature_name)
				plt.legend(loc='upper right')
			else: idx = idx-1
		fig.tight_layout() 
		plt.savefig("./{0}/all_hist_grouped.png".format(folder), bbox_inches='tight')
		plt.show()


	def plot_corr():
		''' Plotting correlations'''
		folder = "corr_{0}".format(dataset_name)
		if not os.path.exists(folder):
			os.makedirs(folder)
		if c_flag == True:
			df.drop(['target'],axis=1)
			number = len(df.columns)-1
		else: number = len(df.columns)
		cor = df.corr()
		fig = plt.figure(figsize=(11, 11))
		plt.imshow(cor, interpolation='nearest')
		
		im_ticks = range(number)
		plt.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
		mask = np.zeros_like(cor)
		mask[np.triu_indices_from(mask)] = True

		plt.xticks(im_ticks, df.columns,  rotation=45)
		plt.yticks(im_ticks, df.columns)
		for i in range(number):
			for j in range(number):
				text = plt.text(j, i, (cor.iloc[i, j]).round(2), ha="center", va="center", color="w")
		plt.colorbar()

		plt.savefig(("./{0}/{1}.png".format(folder,dataset_name)), bbox_inches='tight')
		plt.close('all')


	def plot_scatter(f1, f2):
		'''Scatter for each pair of features'''
		folder = "scatter_{0}".format(dataset_name)
		if not os.path.exists(folder):
			os.makedirs(folder)
		
		mean_f1 = np.mean(df[f1])
		mean_f2 = np.mean(df[f2])
		fig = plt.figure()
		plt.xlabel(f1)
		plt.ylabel(f2)

		if c_flag == True:
			for i in range(len(gr["data"])):
				data_gr = gr["data"][i]
				label_gr = gr["labels"][i]
				x = data_gr.loc[:,f1]
				y = data_gr.loc[:,f2]
				plt.scatter(x, y, label=label_gr)
		else:
			x = df.loc[:,f1]
			y = df.loc[:,f2]
			plt.scatter(x, y)
		plt.scatter(mean_f1, mean_f2, color='g', marker='D', label='mean value')
		plt.legend(loc='upper right')
		plt.savefig(("./{0}/{1}-{2}.png".format(folder, f1.replace('/','-'), f2.replace('/','-'))), bbox_inches='tight')
		plt.close('all')


	def plot_scatter_3d(f1, f2, f3):
		"3D scatter "
		folder = "scatter_{0}".format(dataset_name)
		if not os.path.exists(folder):
			os.makedirs(folder)
		
		fig=plt.figure(figsize=(11, 6), dpi=100)
		ax = fig.add_subplot(111, projection='3d')

		if c_flag == True:
			for i in range(len(gr["data"])):
				data_gr = gr["data"][i]
				label_gr = gr["labels"][i]
				x = data_gr.loc[:,f1]
				y = data_gr.loc[:,f2]
				z = data_gr.loc[:,f3]
				ax.scatter(x, y, z, label=label_gr)
		else:
			x = df.loc[:,f1]
			y = df.loc[:,f2]
			z = df.loc[:,f3]
			ax.scatter(x, y, z)

		ax.set_xlabel(f1)
		ax.set_ylabel(f2)
		ax.set_zlabel(f3)
		ax.legend(loc='upper right')
		plt.savefig(("./{0}/3D_{1}-{2}-{3}.png".format(folder, f1.replace('/','-'), f2.replace('/','-'), f3.replace('/','-'))))
		#plt.show()
		plt.close('all')


	def plot_box():
		'''Box plot for each feature'''
		if c_flag == True:

			folder = "box_{0}".format(dataset_name)
			if not os.path.exists(folder):
				os.makedirs(folder)
			columns = df.columns
			
			for i in range(len(columns)-1):
				trace = []
				for j in range(len(gr["data"])):
					data_gr = gr["data"][j]
					label_gr = gr["labels"][j]
					c = "rgb(" + str(50*j+128) + ", " + str(128+j) + ", " + str(128+j*50) + ")"
					trace.append(go.Box(
						y=data_gr[columns[i]],
						name = label_gr,
						boxpoints = 'suspectedoutliers',
						marker = dict(
						color = c,
						outliercolor = 'rgba(219, 64, 82, 0.6)'
						)
					))
				data = trace #[trace0, trace1]
				layout = go.Layout(
				yaxis=dict(
					title=columns[i],
					zeroline=False
					),
					showlegend = True,
					height = 700,
					width = 1300,
					title='Box plot grouped by Class(target)'
					#boxmode='group'
				)
				fig = go.Figure(data=data, layout=layout)
				plot(fig, filename="./{0}/box_plot_{1}.html".format(folder,columns[i].replace('/','-')), auto_open=False)


	def plot_3d_clustering (f1, f2, f3):
		'''Plotting 3D cluster scatter'''	
		if c_flag == True:
			folder = "3D_{0}".format(dataset_name)
			if not os.path.exists(folder):
				os.makedirs(folder)

			clustered_data = []

		
			for i in range(len(gr["data"])):
				data_gr = gr["data"][i]
				label_gr = gr["labels"][i]
				c = "rgb(" + str(50*i+128) + ", " + str(128+i) + ", " + str(128+i*50) + ")"
				cc = "rgb(" + str(50*i+50) + ", " + str(190+i*6) + ", " + str(200+i*50) + ")"
				xx = data_gr.loc[:,f1]
				yy = data_gr.loc[:,f2]
				zz = data_gr.loc[:,f3]
				scatter = dict(
					mode = "markers",
					name = label_gr,
					type = "scatter3d",    
					x = data_gr.loc[:,f1], y = data_gr.loc[:,f2], z = data_gr.loc[:,f3],
					marker = dict( size=2+i*2, color=cc )
				)
				clustered_data.append(scatter)
				cluster = dict(
					alphahull = 7,
					name = label_gr,
					opacity = 0.1,
					type = "mesh3d",    
					x = xx, y = yy, z = zz,
					color = c
				)
				clustered_data.append(cluster)
			layout = dict(
				title = '3d point clustering',
				scene = dict(
					xaxis = dict( zeroline=False, title=f1 ),
					yaxis = dict( zeroline=False, title=f2 ),
					zaxis = dict( zeroline=False, title=f3 ),
				)
			)
			fig = dict( data=clustered_data, layout=layout )
			plot(fig, filename="./{0}/3D_{1}_{2}_{3}.html".format(folder,f1.replace('/','-'),f2.replace('/','-'),f3.replace('/','-')), auto_open=False)

	return plot_3d_clustering, find_mean_std, plot_box, plot_histograms, plot_histograms_grouped, plot_scatter_3d, plot_scatter, plot_corr

def set_data_analyse(f1, f2):
	X = np.empty(shape=[len(dataset.data), 2])
	y = np.empty(shape=[len(dataset.data),])
	l = []
	k = 0
	for j, c in enumerate(dataset.feature_names):
		if dataset.feature_names[j] == f1 or dataset.feature_names[j] == f2: 
			for i, s in enumerate(dataset.data):
				X[i,k] = dataset.data[i,j]
				y[i] = dataset.target[i]
			l.append(dataset.feature_names[j])
			k = k+1

	X[:,0] = preprocessing.normalize([X[:,0]])
	X[:,1] = preprocessing.normalize([X[:,1]])
	return X, y, l

def plot_results_2D(X_t, y_t, l, name):
	# Create color maps
	folder = "prediction_results_{0}".format(dataset_name)
	if not os.path.exists(folder):
		os.makedirs(folder)
	
	cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA'])
	cmap_bold = ListedColormap(['#FF0000', '#0000FF', '#00FF00'])

	# Calculating prediction desicion mesh based on our algorithm
	x_min, x_max = X_t[:, 0].min() - 0.01, X_t[:, 0].max() + 0.01
	y_min, y_max = X_t[:, 1].min() - 0.01, X_t[:, 1].max() + 0.01
	xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.0001), np.arange(y_min, y_max, 0.0001))
	
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	
	fig = plt.figure()
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.xlabel(l[0])
	plt.ylabel(l[1])	
	
	# Plotting prediction desicion mesh based on our algorithm
	plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
	
	# Plot testing points to check whether they are inside the predicted class 
	plt.scatter(X_t[:,0],X_t[:,1], c=y_t, cmap=cmap_bold, edgecolor='k')

	#plt.legend((X_test[:,0], X_test[:,1]), (dataset.target_names[0], dataset.target_names[1]), loc='lower right')
	plt.title = name
	plt.savefig(("./{0}/{1}_{2}_{3}.png".format(folder, name, l[0].replace('/','-'), l[1].replace('/','-'))), bbox_inches='tight')
	#plt.show()
	plt.close('all')

#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------

# Assigning dataset name to a local variable
dataset_name = parser_assign()

#Loading dataset from sklearn
dataset, classification_flag = load_data(dataset_name) 
print('Classification flag value: ', classification_flag)

# Transrferring sklearn dataset to Data Frame
data, grouped = read_data(dataset['data'], dataset['feature_names'], dataset['target'])
call_3d_clustering, mean_std, box, histograms, histograms_grouped, scatter_3d, scatter, corr = all_functions(classification_flag, data, grouped) 

# Calculating summary statistics
mean_std()

# Plotting histograms
'''print('\n Plotting all histograms into one figure')						#Plotting one histogram for all the features
histograms()
if classification_flag == True:
	print('\n Plotting all histograms into one figure grouped by target')#Plotting one histogram for all the features grouped by diagnosis
	histograms_grouped()'''


#Plotting Box plot
'''print('\n Plotting box plots')
box()'''


# Plotting correlations heatmap
print('\n Plotting correlation hitmap into /corr/ ')
corr()	


# Plotting scatter
'''for i in range(len(data.iloc[0])-1):
	j = 1
	for j in range((i+j),len(data.iloc[0])-1):
		col_name1 = data.iloc[:,i].name
		col_name2 = data.iloc[:,j].name
		print('\n Plotting scatter of ', col_name1, 'and ', col_name2)
		scatter(col_name1, col_name2)'''

	
#Plotting 3D scatter and clustering for custom features
'''if dataset_name == 'breast_cancer':
	print('\n Plotting 3D scatters')
	scatter_3d('mean concave points', 'mean symmetry', 'mean compactness')
	scatter_3d('mean concave points', 'mean smoothness', 'mean compactness')
	scatter_3d('mean concave points', 'mean perimeter', 'mean compactness')
	print('\n Plotting 3D scatters with clustering')
	call_3d_clustering ('mean concave points', 'mean symmetry', 'mean compactness')
	call_3d_clustering ('mean concave points', 'mean smoothness', 'mean compactness')
	call_3d_clustering ('mean concave points', 'mean perimeter', 'mean compactness')
if dataset_name == 'boston':
	print('\n Plotting 3D scatters')
	scatter_3d('RM', 'LSTAT', 'DIS')'''

#---------------------------------------------------------------------------

#-----------CLASSIFICATION ANALYSIS-----------------------------------------

#---------------------------------------------------------------------------
if dataset_name == 'iris':
	feature1 = 'petal length (cm)'
	feature2 = 'petal width (cm)'
if dataset_name == 'breast_cancer':
	feature1 = 'mean concave points'
	feature2 = 'mean perimeter'
if dataset_name == 'wine':
	feature1 = 'proline'
	feature2 = 'od280/od315_of_diluted_wines'
flag = False

if classification_flag == True:

	# Performing principal component analysis (PCA)
	#print('\nPerforming PCA')
	from sklearn.decomposition import PCA
	pca = PCA(n_components=2)
	proj = pca.fit_transform(dataset.data)
	plt.scatter(proj[:, 0], proj[:, 1], c=dataset.target) 
	plt.colorbar() 
	plt.title = 'PCA'
	#plt.show()
	plt.close('all')



	# Performing GaussianNB on all the features
	print('/////////////////////////////////////////////')
	print('Performing GaussianNB on all the features\n')
	X = dataset.data
	y = dataset.target

	X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=0)
	#print('X dataset: ', X.shape, 'y targets: ', y.shape, 'train data shape: ', X_train.shape, 'test data shape: ', X_test.shape)

	clf = GaussianNB()
	clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)

	print('GaussianNB score: ', metrics.f1_score(y_test,y_pred,average="macro"))
	print('cross_val_score mean: ', np.mean(cross_val_score(clf, X, y, cv=5)))
	print(metrics.confusion_matrix(y_test, y_pred))
	#print(metrics.classification_report(y_test, y_pred))

	# Performing Gaussian on two chosen features
	#if dataset_name == 'breast_cancer':
	print('/////////////////////////////////////////////')
	print('Performing GaussianNB on 2 features\n')
	X, y, features = set_data_analyse(feature1, feature2)
	classifier_name = 'GaussianNB'
		
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=0)
	#print('X_train_f1: ', np.max(X_train[:,0]), np.min(X_train[:,0]), features[0])
	#print('X_train_f2: ', np.max(X_train[:,1]), np.min(X_train[:,1]), features[1])
	#print('X_train size: ', X_train.shape, 'y_train size: ', y_train.shape)

	#finding best parameters for SVC
	'''from sklearn.model_selection import GridSearchCV
	print("Fitting the classifier to the training set")
	param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['rbf', 'linear']}
	clf = GridSearchCV(SVC(class_weight='balanced'), param_grid)
	clf = clf.fit(X_train, y_train)
	print("Best estimator found by grid search:")
	print(clf.best_estimator_)'''

	clf = GaussianNB().fit(X_train, y_train)
	plot_results_2D(X_test, y_test, features, classifier_name)
	y_pred = clf.predict(X_test)
	print('GaussianNB on 2 features score: ', metrics.f1_score(y_test,y_pred,average="macro"))
	print('cross_val_score mean: ', np.mean(cross_val_score(clf, X, y, cv=5)))
	print(metrics.confusion_matrix(y_test, y_pred))
	#print(metrics.classification_report(y_test, y_pred))

	

	# Performing SVC on all the features
	#if classification_flag == True:
	print('/////////////////////////////////////////////')
	print('Performing SVC on all the features\n')
	from sklearn.svm import SVC
	X = dataset.data
	y = dataset.target
	
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=0)
	clf = SVC(kernel='linear', gamma='auto').fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	print('SVC score: ', metrics.f1_score(y_test,y_pred,average="macro"))
	print('cross_val_score mean: ', np.mean(cross_val_score(clf, X, y, cv=5)))
	print(metrics.confusion_matrix(y_test, y_pred))
	#print(metrics.classification_report(y_test, y_pred))

	# Performing SVC on two chosen features
	#if dataset_name == 'breast_cancer':
	print('/////////////////////////////////////////////')
	print('Performing SVC on 2 features\n')
	X, y, features = set_data_analyse(feature1, feature2)
	classifier_name = 'SVC'
	
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=0)
	#print('X_train_f1: ', np.max(X_train[:,0]), np.min(X_train[:,0]), features[0])
	#print('X_train_f2: ', np.max(X_train[:,1]), np.min(X_train[:,1]), features[1])
	#print('X_train size: ', X_train.shape, 'y_train size: ', y_train.shape)

	#finding best parameters for SVC
	'''from sklearn.model_selection import GridSearchCV
	print("Fitting the classifier to the training set")
	param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['rbf', 'linear']}
	clf = GridSearchCV(SVC(class_weight='balanced'), param_grid)
	clf = clf.fit(X_train, y_train)
	print("Best estimator found by grid search:")
	print(clf.best_estimator_)'''

	clf = SVC(C=100, kernel='rbf', gamma='auto').fit(X_train, y_train)
	plot_results_2D(X_test, y_test, features, classifier_name)
	y_pred = clf.predict(X_test)
	print('SVC on 2 features score: ', metrics.f1_score(y_test,y_pred,average="macro"))
	print('cross_val_score mean: ', np.mean(cross_val_score(clf, X, y, cv=5)))
	print(metrics.confusion_matrix(y_test, y_pred))
	#print(metrics.classification_report(y_test, y_pred))

	
	
	# Performing KNeighborsClassifier on all the features
	#if classification_flag == True:
	print('/////////////////////////////////////////////')
	print('Performing KNeighborsClassifier on all the features\n')
	X = dataset.data
	y = dataset.target

	X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=0)
	#print('X dataset: ', X.shape, 'y targets: ', y.shape, 'train data shape: ', X_train.shape, 'test data shape: ', X_test.shape)
	
	clf = KNeighborsClassifier(n_neighbors=7).fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	
	#for n in range(1,11):
	#	clf = KNeighborsClassifier(n_neighbors=n).fit(X_train,y_train)
	#	y_pred = clf.predict(X_test)
	#	print('KNeighborsClassifier with {0} neighbors score: '.format(n), metrics.f1_score(y_test,y_pred,average="macro"))

	print('KNeighborsClassifier score: ', metrics.f1_score(y_test,y_pred,average="macro"))
	print('cross_val_score mean: ', np.mean(cross_val_score(clf, X, y, cv=5)))
	print(metrics.confusion_matrix(y_test, y_pred))
	#print(metrics.classification_report(y_test, y_pred))

	# Performing KNeighborsClassifier for the two chosen columns
	#if dataset_name == 'breast_cancer':
	print('/////////////////////////////////////////////')
	print('Performing KNeighborsClassifier on 2 features\n')
	X, y, features = set_data_analyse(feature1, feature2)
	classifier_name = 'KN'

	X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=0)
	
	clf = KNeighborsClassifier(n_neighbors=4).fit(X_train,y_train)
	plot_results_2D(X_test, y_test, features, classifier_name)
	y_pred = clf.predict(X_test)

	'''for n in range(1,11):
		clf = KNeighborsClassifier(n_neighbors=n).fit(X_train,y_train)
		y_pred = clf.predict(X_test)
		print('KNeighborsClassifier score: ', 'k = ', n, ': ', metrics.f1_score(y_test,y_pred,average="macro"))
		print(metrics.confusion_matrix(y_test, y_pred))
		print('KNeighborsClassifier with {0} neighbors score: '.format(n), metrics.f1_score(y_test,y_pred,average="macro"))'''

	print('KNeighborsClassifier score: ', metrics.f1_score(y_test,y_pred,average="macro"))
	print('cross_val_score mean: ', np.mean(cross_val_score(clf, X, y, cv=5)))
	print(metrics.confusion_matrix(y_test, y_pred))
	#print(metrics.classification_report(y_test, y_pred))
