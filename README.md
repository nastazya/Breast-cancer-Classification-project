# final_project
Loading multiple datasets from scikit-learn, visualizing them, performing and comparing basic classification algorithms


| Name | Date |
|:-------|:---------------|
| Anastasia Haswani | March 23, 2019 |

-----

## Research Question

Using breast cancer Wisconsin dataset explore and evaluate multiple approaches performance of three machine learning algorithms (GaussianNB, Nearest Neighbours and SVC)  in  order  to  choose  the  one  which  will fit best for breast cancer diagnose.

### Abstract

Breast cancer dataset contains features of a breast mass and used to predict the diagnosis: malignant or benign. For training purposes, we used 10 mean features for general analysis and visualization of the data. Then we compared the performance of classificators of our choice while using two approaches: predicting in 2-D space using 2 "best" features and predicting using all of the 30 features.  
Based on resulting scores we found out that in our case all of the classificators showed better results in 30-dimensional normalized space that brings us to a conclusion that 2 features are not enough to predict medical diagnosis. 

### Introduction

The dataset used in our project is a training Scikit-learn dataset. It can be downloaded from their page https://scikit-learn.org/stable/datasets/index.html#toy-datasets or imported from `sklearn` library. It contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe the characteristics of the cell nuclei present in the image. The mean, standard error, and “worst” or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. All of the features are linearly separable and used to predict the diagnosis: malignant or benign.
[Source here](https://scikit-learn.org/stable/datasets/index.html#breast-cancer-wisconsin-diagnostic-dataset)

### Methods

Here are the short description of the classificators we used:

**Nearest Neighbors**: Finding majority vote of the nearest neighbors of each point: a query point is assigned the data class which has the most representatives within the nearest neighbors of the point. [Pseudocode (Statistical setting and Algorithm)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm#Dimension_reduction)

**Gaussian NB**: Model that assign class labels to problem instances, represented as vectors of feature values, where the class labels are drawn from the training set (assumption is made that the continuous values associated with each class are distributed according to a Gaussian distribution). [Pseudocode](https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes)

**SVC**: Building a n-1 dimentional plane maximizing the margin between two classes. [Pseudocode (Section 1.4.7.1)](https://scikit-learn.org/stable/modules/svm.html#mathematical-formulation)

Nearest Neighbors and Gaussian NB were chosen for their simplicity. SVC was chosen as it works well high dimensional spaces, that fits our case. 

### Results

Below is the histogram of the features grouped by diagnosis, and the correlation heatmap. When looking at the histogram we notice that the distribution of malignant cells appears to be shifted to the right and shaped differently for several of the features. For 2-D space predictions `mean concave points` and  `mean perimeter` were chosen because they are well related but also we've noticed that they don't have many outliers (see scatter plot). 

![hist](https://user-images.githubusercontent.com/46948881/54732376-c75e3500-4b69-11e9-8e85-e1dbfd2cb80d.jpg)

![corr+scatter](https://user-images.githubusercontent.com/46948881/54730580-169f6800-4b60-11e9-9092-d93d86202518.jpg)

Before starting with prediction analysis we've normalized our data since the range of our feature values varies widely. 
At first, we made predictions on 2 features. After optimizing the hyperparameters (`k=5`, `weights='uniform'` for KNN, `C=100`, `kernel='rbf'`, `gamma='scale'` for SVC) we got these results:

![2D_all_num_folds=10](https://user-images.githubusercontent.com/46948881/54763958-46d12000-4bcd-11e9-9704-8172334f9105.png)

 Next step was to do the same thing on all of the 30 features. We optimized the hyperparameters to `k=1` for KNN and `C=100`, `kernel='rbf'`, `gamma='scale'` for SVC. Below is the comparison table with the relsulting values.
 
 
 
In order to visualize all of our steps and compare our approaches, we built the comparison box plots of cross-validation scores for each of the algorithms performing 2-D space and 30-D space predictions before and after optimization of the algorithms. The 10-fold cross validation procedure was used to evaluate each algorithm, importantly configured with the same random seed to ensure that the same splits to the training data are performed and that each algorithms is evaluated in precisely the same way.
 
 

### Discussion
Brief (no more than 1-2 paragraph) description about what you did. Include:

- interpretation of whether your method "solved" the problem
- suggested next step that could make it better.

### References

https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification

https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes

https://scikit-learn.org/stable/modules/svm.html#mathematical-formulation

https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm#Dimension_reduction

https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/

https://github.com/gkiar/cebd1160_project_template/tree/gkexample

-------
