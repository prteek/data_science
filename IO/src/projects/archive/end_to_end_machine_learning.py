# author           : Prateek
# email            : prateekpatel.in@gmail.com
# description      : Exercise in end-to-end machine learning project

import os
import tarfile
import urllib
from datetime import datetime as DT

# In[2]:


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

print(HOUSING_URL)


# In[3]:


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# In[4]:


import pandas as pd


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# In[5]:


fetch_housing_data()

housing = load_housing_data(HOUSING_PATH)
housing.head()


# ### Take a Quick look at the Data Structure

# In[6]:


housing.info()


# In[7]:


housing.ocean_proximity.value_counts()  # useful for non numeric variables


# In[8]:


housing.describe()  # useful for numeric variables


# In[9]:

import matplotlib.pyplot as plt

housing.hist(bins=50, figsize=(20, 15))
plt.show()


# In[10]:


from sklearn.model_selection import train_test_split
import numpy as np

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# In[11]:


housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5],
)


# In[12]:


housing.income_cat.hist()


# In[13]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing.income_cat):
    strat_train_set = housing.iloc[train_index]
    strat_test_set = housing.iloc[test_index]


print(strat_train_set.income_cat.value_counts() / len(strat_train_set), "\n")
print(housing.income_cat.value_counts() / len(housing))


# In[14]:


strat_train_set.drop("income_cat", axis=1, inplace=True)
strat_test_set.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()

housing.head()


# ### Visualising Geographical data

# In[15]:


housing.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    alpha=0.1,
    s=housing["population"] / 100,
    label="population",
    c="median_house_value",
    cmap=plt.get_cmap("jet"),
    colorbar=True,
)
plt.legend()
plt.grid()
plt.show()


# ### Looking for Correlations

# In[16]:


corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)


# In[17]:


from pandas.plotting import scatter_matrix

attributes = [
    "median_house_value",
    "median_income",
    "total_rooms",
    "housing_median_age",
]
scatter_matrix(housing[attributes], figsize=(12, 8), alpha=0.1)
plt.show()


# In[18]:


housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
plt.show()


# ### Experimenting with Attribute Combinations

# In[19]:


housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]


corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)


# ### Data cleaning

# In[20]:


housing = strat_train_set.drop(
    "median_house_value", axis=1
)  # drop is automatically creating a copy of dataframe
housing_labels = strat_train_set["median_house_value"].copy()

# Total bedrooms has missing values so either remove them or replace them with some value

housing.dropna(subset=["total_bedrooms"], inplace=False)

# or

median = housing[
    "total_bedrooms"
].median()  # Use the same median for Test set and beyond

housing["total_bedrooms"].fillna(median, inplace=True)
# Inplace is needed to actually change the dataframe itself rather than its copy


# In[21]:


from sklearn.impute import SimpleImputer  # Scikit-Learn 0.20+

imputer = SimpleImputer(strategy="median")

# Create dataframe with only numerical features to calculate and apply median to all features
housing_num = housing.drop("ocean_proximity", axis=1)

imputer.fit(housing_num)

# print('medians of all features:', imputer.statistics_)

X = imputer.transform(housing_num)
print(type(X))

housing_tr = pd.DataFrame(X, columns=housing_num.columns)


# ### Handling text and categorical attributes

# In[22]:


housing_cat = housing[["ocean_proximity"]].copy()  # use double brackets for array
# or reshape later using housing_cat.values.reshape(-1,1)

from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(ordinal_encoder.categories_)
housing_cat_encoded[:10]


# In[23]:


# One issue with above is ML algorith will assume two nearby values are more similar than two distant values
#  To avoid this we can use one-hot encoding

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(type(housing_cat_1hot))
print(housing_cat_1hot.toarray())

# This is a sparse scipy array. Very handy when there are thousands of categories to be encoded.
# Sparse array takes up much less memory


# ### Custom Transformers

# In[24]:


from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

# sklearn objects acre consistent as in they output numpy array. We'll do the same


class CombinedAttributesAdder(
    BaseEstimator, TransformerMixin
):  # no args or **kwargs needed because of BaseEstimator
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else needed to be done

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[
                X, rooms_per_household, population_per_household, bedrooms_per_room
            ]  #  numpy array
        else:
            return np.c_[
                X, rooms_per_household, population_per_household
            ]  # numoy array


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)

housing_extra_attributes = attr_adder.transform(housing.values)
cols = np.append(housing.columns.values, [["rph", "pph"]])
housing_extra_attributes = pd.DataFrame(housing_extra_attributes, columns=cols)

housing_extra_attributes.head()


# ### Feature Scaling
# 1. Min-Max Scaling: Bounds values between 0-1 (required by some ML algorithms) but strongly affected by outlier (squishes values to very small values)
# 2. Standardisation: Does not bound and is not much affected by outlier

# In[25]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(with_mean=True, with_std=True, copy=True)
scaled_data = scaler.fit_transform(housing_num)
scaled_data = pd.DataFrame(scaled_data, columns=housing_num.columns)
scaled_data.head()


# ### Transformation Pipelines
#
# The constructore takes list of name/estimator pairs defining a sequence of steps.
# All but the last estimator must be transformers must be transformers (i.e., they must have fit_transform() method).
# The names must not contain double underscores "__"
#

# In[26]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", CombinedAttributesAdder()),
        ("std_scaler", StandardScaler(copy=True, with_mean=True, with_std=True)),
    ]
)

housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[27]:


# But to apply pipelines separately to numerical and categorical attributes lets define an object
# that selects specified attributes first and drops the rest and then use this along with transformers in pipeline.
# This allows to directly input the entire DataFrame instead of selecting numerical attributes first and then using Pipeline


from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.attribute_names].values  # output a numpy array


num_attributes = housing_num.columns.values
cat_attributes = ["ocean_proximity"]

num_pipeline = Pipeline(
    [
        ("selector", DataFrameSelector(num_attributes)),
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", CombinedAttributesAdder()),
        ("std_scaler", StandardScaler(copy=True, with_mean=True, with_std=True)),
    ]
)

cat_pipeline = Pipeline(
    [("selector", DataFrameSelector(cat_attributes)), ("cat_encoder", OneHotEncoder())]
)


housing_num_tr = num_pipeline.fit_transform(housing)
housing_cat_tr = cat_pipeline.fit_transform(housing)


# In[28]:


# To combine these two pipelines

from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(
    transformer_list=[("num_pipeline", num_pipeline), ("cat_pipeline", cat_pipeline)]
)

housing_prepared = full_pipeline.fit_transform(housing)

housing_prepared.shape


# ### Training and evaluating on the Traning set

# In[29]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# To avoid less number of resulting columns after OneHotEncoding because we only want a subset of housing,
# as not all possible values of 'ocean_proximity' may be in the subset, use prepared data itself

some_data_prepared = housing_prepared[:5]
some_labels = housing_labels.iloc[:5]

print("Predictions: ", lin_reg.predict(some_data_prepared))
print("Actual labels: ", some_labels.values, "\n")


from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("Linear Regression rmse: ", lin_rmse)

# This is very high prediction error and model is underfitting. Let's try another model

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("Decision tree rmse: ", tree_rmse)

# 0 error! means model is severyly over fitting. A good way is to use cross validation using validation sets


# ### Better Evaluation using Cross-Validation

# In[30]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10
)
tree_rmse_scores = np.sqrt(
    -scores
)  # since cross val uses utility function instead of cost function use -ve sign


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", np.mean(scores))
    print("Std:", np.std(scores))


display_scores(tree_rmse_scores)

lin_scores = cross_val_score(
    lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10
)

lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
# The prediction error is worse than Linear Regression possibly due to severe overfitting


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
# forest_reg.fit(housing_prepared, housing_labels)

# forest_scores       = cross_val_score(forest_reg, housing_prepared, housing_labels,
#                                      scoring='neg_mean_squared_error', cv=10)

# forest_rmse_scores  = np.sqrt(-forest_scores)

# display_scores(forest_rmse_scores)

## Random Forest performs the best among models so far evaluated
## let's save this

from sklearn.externals import joblib

# joblib.dump(forest_reg, 'forest_regressor_default.joblib')

## and later re-use it
# forest_reg_model = joblib.load('forest_regressor_default.joblib')


# ### Grid Search

# In[32]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(
    forest_reg, param_grid=param_grid, cv=5, scoring="neg_mean_squared_error"
)

tic = DT.now()
grid_search.fit(housing_prepared, housing_labels)
print("Grid Search:", DT.now() - tic)

# In[52]:


# Get the best parameters

print(grid_search.best_params_, "\n")

#  Get the best estimators
print(grid_search.best_estimator_)

# If GridSearchCV is initialised with refit=True (default) then once it finds the best combination
# it will retrain the best model on the entire training set, this will likely improve performance.


# In[54]:


# Evaluation scores can be queried by

cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# ### Randomized Search
# If the search space of parameters is large then use **RandomizedSearchCV** as this will randomise the search space and same time. It can be used much in the same way as GridSearchCV

# ### Analyse the best model and its error

# In[66]:


feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)

extra_attribs = ["rooms_per_hhold", "population_per_hhold", "bedrooms_per_hhold"]
cat_encoder = cat_pipeline.named_steps["cat_encoder"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])

attributes = list(num_attributes) + extra_attribs + cat_one_hot_attribs

sorted(zip(feature_importances, attributes), reverse=True)

# We may drop the features with less importance


# ### Evaluating on Test data

# In[68]:


final_model = grid_search.best_estimator_

x_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

x_test_prepared = full_pipeline.transform(x_test)
final_predictions = final_model.predict(x_test_prepared)

final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))

print(final_rmse, "This may be worse than Validation error")
