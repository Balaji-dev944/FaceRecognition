"""
===================================================
Faces recognition example using eigenfaces and SVMs
===================================================

The dataset used in this example is a preprocessed excerpt of the
"Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

.. _LFW: http://vis-www.cs.umass.edu/lfw/

Expected results for the top 5 most represented people in the dataset:

================== ============ ======= ========== =======
                   precision    recall  f1-score   support
================== ============ ======= ========== =======
     Ariel Sharon       0.67      0.92      0.77        13
     Colin Powell       0.75      0.78      0.76        60
  Donald Rumsfeld       0.78      0.67      0.72        27
    George W Bush       0.86      0.86      0.86       146
Gerhard Schroeder       0.76      0.76      0.76        25
      Hugo Chavez       0.67      0.67      0.67        15
       Tony Blair       0.81      0.69      0.75        36

      avg / total       0.80      0.80      0.80       322
================== ============ ======= ========== =======

"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
#from svmutil import svm_problem,svm_parameter,svm_model
from libsvm.svmutil import svm_problem,svm_train,svm_parameter,svm_predict
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA







###############################################################################
# Download the data, if not already on disk and load it as numpy arrays

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)


# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
print("X shape",X.shape)
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
print(y.shape)
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


###############################################################################
#Split into a training set and a test set using a stratified k fold

#split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

print("No of Training samples=",X_train.shape[0])
print("No of Test Samples",X_test.shape[0])

################################################################################
## Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
## dataset): unsupervised feature extraction / dimensionality reduction
n_components = 170

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))

pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)


eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

#
################################################################################
## Train a SVM classification model
#
print("Fitting the classifier to the training set")
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
prob = svm_problem(y_train,X_train_pca)
param = svm_parameter("-q")
param.kernel='rbf'
#param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
 #             'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
param.C=32
param.gamma=0.0001
print("Pass")
#parameters = GridSearchCV( param_grid)
model= svm_train(prob,param)
#clf = clf.fit(X_train_pca, y_train)

#print("Best estimator found by grid search:")
#print(m.best_estimator_)
y_pred, pred_acc, pred_val = svm_predict(y_test,X_test_pca,model)

################################################################################
## Quantitative evaluation of the model quality on the test set


print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


print("Predicting people's names on the test set")
################################################################################
## Qualitative evaluation of the predictions using matplotlib
#
def plot_gallery(images, titles, h, w, n_row=4, n_col=8):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.2 * n_col, 2.2 * n_row))
    plt.subplots_adjust(bottom=0, left=0.1, right=.99, top=.95, hspace=.25)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=6)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[int(y_pred[i])]
    true_name = target_names[int(y_test[i])]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range (len(y_pred))]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

#eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
#plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()
