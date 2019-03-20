# final_project
Loading multiple datasets from scikit-learn, visualizing them and performing basic classification algorithms


| Name | Date |
|:-------|:---------------|
| Anastasia Haswani | March 23, 2019 |

-----

### Resources
Your repository should include the following:

- Python script for your analysis
- Results figure/saved file
- Dockerfile for your experiment
- runtime-instructions in a file named RUNME.md

-----

## Research Question

Using breast cancer Wisconsin dataset explore and evaluate multiple approaches performance of three machine learning algorithms (GaussianNB, nearest neighbour and SVC)  in  order  to  choose  the  one  which  will fit best for breast cancer diagnose.

### Abstract

Breast cancer dataset contains features of a breast mass and used to predict the diagnosis: malignant or benign. For training purposes, we used 10 mean features for general analysis and visualization of the data. Then we compared the performance of classificators of our choice while using two approaches: predicting in 2-D space using 2 "best" features and predicting using all of the 30 features.  
Based on resulting scores we found out that in our case all of the classificators showed better results in 30-dimensional normalized space that brings us to a conclusion that 2 features are not enough to predict medical diagnosis. 

### Introduction

The dataset used in our project is a training Scikit-learn dataset. It can be downloaded from their page [https://scikit-learn.org/stable/datasets/index.html#toy-datasets] or imported from `sklearn` library. It contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe the characteristics of the cell nuclei present in the image. The mean, standard error, and “worst” or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. All of the features are linearly separable and used to predict the diagnosis: malignant or benign.
(Source: [https://scikit-learn.org/stable/datasets/index.html#breast-cancer-wisconsin-diagnostic-dataset])

### Methods

Brief (no more than 1-2 paragraph) description about how you decided to approach solving it. Include:

- pseudocode for this method (either created by you or cited from somewhere else)
- why you chose this method

### Results

Brief (2 paragraph) description about your results. Include:

- At least 1 figure
- At least 1 "value" that summarizes either your data or the "performance" of your method
- A short explanation of both of the above

### Discussion
Brief (no more than 1-2 paragraph) description about what you did. Include:

- interpretation of whether your method "solved" the problem
- suggested next step that could make it better.

### References
All of the links

-------
