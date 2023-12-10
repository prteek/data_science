# author           : Prateek
# email            : prateekpatel.in@gmail.com
# description      : Exercise in classification models

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.base import clone, TransformerMixin, BaseEstimator
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split



from sklearn.datasets import load_digits

mnist = load_digits()

mnist.keys()



mnist["DESCR"]


X, y = mnist["data"], mnist["target"]
y = y.astype(int)

X.shape, y.shape


# get_ipython().run_line_magic('matplotlib', 'inline')
index = 11
some_digit = X[index]
some_digit_image = some_digit.reshape(8, 8)

plt.figure()
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

print(y[index])



# Create a test set and set it aside before further exploration

# X_train, X_test, y_train, y_test = X[:60000],  X[60000:], y[:60000], y[60000:]
# random_index = np.random.permutation(6000)
# Also randomise the train set data for better cross validation

# X_train, y_train = X_train[random_index], y_train[random_index]

#  To accomodate data in memory
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)




from sklearn.linear_model import SGDClassifier

y_train_5 = y_train == 5
y_test_5 = y_test == 5

sgd_clf = SGDClassifier(random_state=42)

sgd_clf.fit(X_train, y_train_5)

sgd_clf.predict([some_digit])



# Custom made cross_val_score

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)

# Clonining should be done inside the loop to have a new classifier each time
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_test_fold == y_pred)
    print(n_correct / len(y_pred))


from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")


# This accuracy looks too good. Let's create a classifier that classifies every input as not 5

from sklearn.base import BaseEstimator


class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        return None

    def predict(self, X, y=None):
        return np.zeros((len(X), 1), dtype=bool)


never_5_clf = Never5Classifier()

print(cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy"))
print("Not too bad for a dumb classifier")

print(sum(y_train_5) / len(y_train_5), "Only a small proportion of training data are 5")
print("Accuracy is not the preferred performance measure for skewed datasets")



# Confusion Matrix

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# This returns the (clean) prediction for each instance in the training set by treating them in test set
# in different cross validation iterations. Clean means the predictions are by a model which hasn't seen that instance
# while training.

cm = confusion_matrix(y_train_5, y_train_pred)
# Each row is an actual class (negative, positive), each column is a predicted class (negative, positive)
print(cm)



print(
    "Precision TP/(TP+FP) = ", cm[1, 1] / (cm[1, 1] + cm[0, 1])
)  #  Accuracy of 'positive predictions'
# A cheat for precision is to predict only 1 true value and make it accurate so Precision = 1/(1+0)
# To not have this situation we also look at how many of true values are correctly predicted
print(
    "Recall TP/(TP+FN)= ", cm[1, 1] / (cm[1, 1] + cm[1, 0])
)  # What proportion of 'positive Actuals' are correctly predicted
print(
    "F1 score = ",
    2
    / (1 / (cm[1, 1] / (cm[1, 1] + cm[0, 1])) + 1 / (cm[1, 1] / (cm[1, 1] + cm[1, 0]))),
)

# F1 score is the harmonic mean of Precision and recall values. As Harmonic mean places more emphasis on lower values
# therefore, both precision and recall must be high to get good F1 score


from sklearn.metrics import precision_score, recall_score, f1_score

print("Precision score", precision_score(y_train_5, y_train_pred))
print("Recall score", recall_score(y_train_5, y_train_pred))
print("F1 score", f1_score(y_train_5, y_train_pred))

# Precision and recall are complementary i.e. increasing one decreases the other


# Check the score used to make prediction, threshold for making decision is unavailable

y_scores = sgd_clf.decision_function([some_digit])
print(y_scores)

# But we can make our own threshold and use it to compare to score to make decision
threshold = 0
y_some_digit_predict = y_scores > threshold
print(y_some_digit_predict)

threshold = 200000
y_some_digit_predict = y_scores > threshold
print(y_some_digit_predict)

# Increasing the threshold will decrease Recall



# To decide on a value threshold to optimise precision vs recall, first get all the prediction scores
y_scores = cross_val_predict(
    sgd_clf, X_train, y_train_5, cv=3, method="decision_function"
)
# Using 'decision_function' gives the value of score rather than class itself

print(y_scores[:5])




# Now precision and recall can be computed for all possible thresholds

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.figure()
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    plt.grid()
    plt.show()


plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

print(
    "Precision may (or may not) decrease on increasing threshold, Recall will always decrease"
)

plt.figure()
plt.plot(precisions, recalls)
plt.xlabel("precisions")
plt.ylabel("recalls")
plt.grid()
plt.show()
print(
    "Recall falls of quickly after a certain point, \n choose precision where recall is also reasonably high for the application"
)



y_train_pred_90 = y_scores > 200000
# Let's say we want 90% precision then threshold is correspondingly chosen

print("Precision: ", precision_score(y_train_5, y_train_pred_90))
print("Recall: ", recall_score(y_train_5, y_train_pred_90))



# The ROC curve

from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


def plot_roc_curve(fpr, tpr, label=None):
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], "k--")
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()


plot_roc_curve(fpr, tpr)

print("Area under curve (AUC)", roc_auc_score(y_train_5, y_scores))
print("Ideal AUC=1, Random classifier AUC = 0.5")



from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(
    random_state=42
)  # Does not have a decision_function() rather has predict_proba()

y_probas_forest = cross_val_predict(
    forest_clf, X_train, y_train_5, cv=3, method="predict_proba"
)

y_scores_forest = y_probas_forest[:, 1]  # score = probabilities of positive class

fpr_forest, tpr_forest, thresholds = roc_curve(y_train_5, y_scores_forest)


plt.figure()
plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()


print("Area under curve (AUC)", roc_auc_score(y_train_5, y_scores_forest))



# Multi Class classification

# Scikit detects when we try to use a binary classification algorithm for a multiclass classification task.
# It automatically runs One vs Rest (except for SVM classifiers where it uses OvO)

sgd_clf.fit(X_train, y_train)  # Using all classes in training
sgd_clf.predict([some_digit])



some_digit_scores = sgd_clf.decision_function([some_digit])
print(some_digit_scores, "\n\n maximum score for class:5")

sgd_clf.classes_



from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
print(ovo_clf.predict([some_digit]))

len(ovo_clf.estimators_)  # 10Choose2 One vs One classifiers for 10 classes = 45



# Random forest classifier has multiclass capability

forest_clf.fit(X_train, y_train)
print(forest_clf.predict([some_digit]))
print(forest_clf.predict_proba([some_digit]))



print(
    "SGDClassifier: ",
    cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy"),
)

# This is good but can be improved by feature scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

print(
    "SGD with scaled features: ",
    cross_val_score(sgd_clf, X_train_scaled, y_train, cv=5, scoring="accuracy"),
)



# Error Analysis

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=5)

conf_mx = confusion_matrix(y_train, y_train_pred)
print(conf_mx)

plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()



row_sums = conf_mx.sum(
    axis=1, keepdims=True
)  # Normal;ise the conf_mx by number of images in each class
norm_conf_mx = conf_mx / row_sums

# Fill diagonals with zeros to keep only the errors

np.fill_diagonal(norm_conf_mx, 0)
plt.figure()
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()


# Obeserve that many other digits are predicted as 8 so column (predicted) for 8 is bright
# Also that 5 and 3 are often confused with each other
# Row for 2 is bright so 2 is confused for other digits in prediction



# EXTRA
def plot_digits(instances, images_per_row=10, **options):
    size = 8
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=plt.cm.binary, **options)
    plt.axis("off")


# Let us observe the confusion between 3 and 5

cl_a, cl_b = 1, 8

X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]


plt.figure(figsize=(8, 8))
plt.subplot(221)
plot_digits(X_aa[:10], images_per_row=5)
plt.subplot(222)
plot_digits(X_ab[:10], images_per_row=5)
plt.subplot(223)
plot_digits(X_ba[:10], images_per_row=5)
plt.subplot(224)
plot_digits(X_bb[:10], images_per_row=5)
plt.show()




# Multilabel classification

from sklearn.neighbors import KNeighborsClassifier

y_train_large = y_train >= 7  # Large numbers in y
y_train_odd = y_train % 2 == 1  # Odd digits in y

y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

knn_clf.predict([some_digit])

