import streamlit as st
import pandas as pd

def run():

    st.title("London housing data modelling")
    st.caption("""This is an exercise in predicting number of bedrooms from a housing dataset.  
More description of the problem and the notebook where actual analysis is performed can be found in this [repo](https://github.com/prteek/bedroom-prediction) """)
    
    st.markdown("""## Data exploration""")

    df = pd.read_csv('https://raw.githubusercontent.com/prteek/bedroom-prediction/main/home_search.data')
    dfna = df.isna().sum().reset_index().iloc[1:]
    dfna[0] = df.shape[0] - dfna[0]
    dfna.columns = ['Column', 'Non-Null Count']
    dfna['Dtype'] = df.dtypes.reset_index()[0].apply(lambda x: x.__str__())
    
    st.table(dfna)
    
    st.markdown("""Looking at missing values in the data we can infer the following:  
1. There are quite a few missing values in **chain_1_hash** (we can keep it along until observing its informativeness else we can drop this column.  
2. For other columns NaN values are a small fraction of total so we can potentially just drop NaN rows from these columns for simplicity
""")
    
    st.markdown("""
### Checking correlation between features (using numeric features only)
We can identify collinearity between features and drop some of them to avoid instabilities in the models later on
"""
    )

    st.image("https://raw.githubusercontent.com/prteek/bedroom-prediction/main/images/pearson_correlation_numeric_features.png")
    
    st.markdown("""- It makes sense that relh2 and relhmax are correlated and so are absh2 and abshmax. For these variables we can potentially just take one and drop the other 
- To a significant degree abshmin happens to be correlated with absh2 and abshmax. We can explore this in slight more detail before making a decision
""")
    
    st.image("https://raw.githubusercontent.com/prteek/bedroom-prediction/main/images/absh2_vs_abshmin.png")
    
    st.markdown("""Looking at the plot it appears that `abshmin` encodes slightly more information than absh2 being bimodal so we can keep this and **drop absh2 and abshmax**. We will also keep `relhmax` and these both should approximately capture the effect of elevation and build height
""")
    
    
    st.markdown("""### Checking informativeness in features""")
    
    st.image("https://raw.githubusercontent.com/prteek/bedroom-prediction/main/images/mutual_info_for_classification_task.png")
    
    st.markdown("""Earlier it was suggested that we may drop `chain_1_hash` because it has many NaN values which can't realistically be imputed. 
However, this column has much information about target variable and instead of losing all that information we prefer to model houses that do not have this information separately (or working with models that can incorporate missing values natively).
This is also important because dropping *NaN* value rows on the basis of `chain_1_hash` removes data about **Detached** houses entirely (blank row on chart).
""")
    
    st.markdown("""### Target analysis
Ideally for the nature of target variable (counts) we would approach the problem as some version of **Poisson regression**.
However, it is important to not have an error of even single bedroom in our predictions means we can frame the problem as a classification problem and try to achieve high accuracy on the classification task.
""")
   
    st.image("https://raw.githubusercontent.com/prteek/bedroom-prediction/main/images/bedrooms_class_imbalance.png")
    
    st.markdown("""Since the distribution of classes being non-uniform a proper train-test split should be used (Stratified for simplicity).  
The test set can be reserved until we arrive at a sufficiently satisfactory model and evaluate final performance on this set.  
Because we will need to do a lot of cross validation we'll not keep the test set size un necessarily large. (we'll split 80%-20%)
""")
    
    st.image("https://raw.githubusercontent.com/prteek/bedroom-prediction/main/images/stratified_train_test_split.png")   
    
    st.markdown("""We appear to have a sensible split of training and test data and can progress to modelling from here.
""")
    
    
    st.markdown("""### Modelling

We can start with a simple model which we can consider as baseline and then build more complex models successively improving the scores

**Assumptions**
- We don't wan't to focus on any particular class we can just assume `accuracy` as a good metric to optimise
- Because the dataset has large number of instances and quite a few features, it is likely that parametric models may underfit the data. Hence we'll start with minimal regularisation
- We will first build a suitable model with `chain_1_hash` included and then move on to model instances that have `chain_1_hash` value missing. If needed we can then merge 2 models into 1 composite 
""")
    
    st.markdown("""### Baseline model
To start with we'll model the problem with a `DummyClassifier` to get a handle on lower bounds of performance and keep track of incremental improvements
""")
    
    st.image("https://raw.githubusercontent.com/prteek/bedroom-prediction/main/images/dummy_classifier_confusion_matrix.png")  
    
    
    st.image("https://raw.githubusercontent.com/prteek/bedroom-prediction/main/images/dummy_classifier_performance.png")
    
    st.markdown("""This model makes reasonable assumption of predicting class instances in the ration in which they appear in traininng data. 
To take into account the stochastic nature of train-test split we'll use `cross_val_score` to evaluate this model.  
Multiple evaluations with `cross_val_score` give us a distribution of accuracy where variance is the effect of randomness in how the data is split.
""")
    
    st.markdown("""### Linear model
* For simplicity we can start with a linear model (Logistic regression). It is preferred since it can account for categorical variables in our data as compared to LDA which would have been an alternate choice since it could account for prior class probabilities more explicitly which could be desirable sometimes.
* For robust estimation of coefficients it is needed that input variables be scaled and PCA be applied to get rid of correlations among features
* Since we know that there are many features and sufficiently large number of training instances, this model will naturally underfit the data hence we do not regularise the model.
""")
    
    
    st.image("https://raw.githubusercontent.com/prteek/bedroom-prediction/main/images/dummy_classifier_vs_logreg.png")
    
    st.markdown("""There is clear improvement over `DummyClassifier` (even accounting for variance due to randomness in data splitting) and this may be due to the fact that the decision boundaries in higher dimensional space are actually simple enough to be captured by linear model. There is however room for much improvement.  
Let's check the bias-variance behavour of the model
""")
    
    st.image("https://raw.githubusercontent.com/prteek/bedroom-prediction/main/images/logreg_learning_curve.png")
    
    st.markdown("""Both the training and CV scores are low albeit close, meaning that our model is underfitting rather than overfitting the training data. Although adding more training instances does not still improve the training score, meaning that we probably have enough number of instances to work with but our features do not have enough predictive power.  
This observation alone prompts us to do some feature engineering.  
One common way is to include some non linearity and some feature interactions in the model by including polynomial transformation of feature vectors.
""")
    
    st.image("https://raw.githubusercontent.com/prteek/bedroom-prediction/main/images/logreg_vs_logreg_poly.png")
    
    st.markdown("""As an experiment polynomial of degree 3 is trialled along with feature interactions, this does not seem to increase the score all that much and so the model structurally cannot improve much unless extensive feature engineering is performed e.g. spline transformations etc.
At this point we can try more complex models which may be less interpretable but more flexible and accurate.
""")
    
    st.markdown("""### Ensemble model
As compared to linear model it is expected that the ensemble model `GradientBoost` should perform better due to inherently capturing complex feature interactions as well correcting on errors by building successively un correlated trees.  
Another benefit with `GBM` is they can handle missing values natively and we do not need to build separate model for missing values in `chain_1_hash`
""")
    
    st.image("https://raw.githubusercontent.com/prteek/bedroom-prediction/main/images/logreg_vs_gradboost.png")
    
    st.image("https://raw.githubusercontent.com/prteek/bedroom-prediction/main/images/gradboost_confusion_matrix.png")
    
    st.markdown("""Model has realistic improvements even accounting for variance in accuracy estimation due to data splits, but it could be further tuned to get more accurate as a first step. We have to identify however if the model is overfitting or likely underfitting the data
""")
    
    
    st.image("https://raw.githubusercontent.com/prteek/bedroom-prediction/main/images/gradboost_learning_curve.png")
    
    st.markdown("""We can see that since the *training curve* is above the *validation curve*, we're significantly overfitting the data using this model.
It can be said that to avoid this overfitting having more samples could help as the curves appear to be slightly converging even when using maximum number of samples.  
This indicates a need to introduce regularisation for better generalising the model fit.  
We can start by decreasing number of stages in the ensemble build and observe the effect.
""")
    
    
    st.image("https://raw.githubusercontent.com/prteek/bedroom-prediction/main/images/gradboost_n_estimators_tuning.png")
    
    st.markdown("""It appears that the *validation scores* start flattenning out after 50 stages and *training score* stronngly starts to diverge from this point so we'll limit the ensemble there.  
Reducing the number of samples used in individual stages also strongly regularises the ensembles so we can experiment with this
""")
    
    st.image("https://raw.githubusercontent.com/prteek/bedroom-prediction/main/images/gradboost_subsample_tuning.png")
    
    st.markdown("""At around `subsample` = 0.4 the capability of ensemble to generalise well saturates, however increasing this value does not result in diverging curves and hence there is no benefit in changing this from default value.  
As a last option we'll tweak `min_samples_leaf`, increasing this value should introduce more regularisation in the system
""")
    
    st.image("https://raw.githubusercontent.com/prteek/bedroom-prediction/main/images/gradboost_min_samples_leaf_tuning.png")
    
    st.markdown("""This largely shows no impact (on *validation score*) and we can conclude that limiting number of stages has already regularised model quite enough. We can look into more granular level now at misclassified instances to see if there is a pattern and if it can be used to correct model in the correct direction.  

At this point it is recommended to use `GradientBoostClassifier` model since it does overfit on training data meaning it has the fleibility to capture patterns in the data. However, with proper generalisation and error investigation this can be limited and *validation* score can be improved.
""")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    