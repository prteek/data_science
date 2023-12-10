import streamlit as st


def run():
    st.title("Fit 500 models in one line with GPU")
    
    st.subheader("Intro")
    st.markdown("""Having done much modelling and data processing one often encounters situations where these
exercises start taking more time than we'd want to.  
Often times people either have a gaming PC/laptop at home or the work PC has some mojo for Graphics intensive processing 
and comes equipped with a GPU. Either case it's a good opportunity to dive into GPU processing and exploring how we could speed up
our workflows with little time investment. 

Note: It is assumed here that the device comes equipped with one of the newer (not newest) Nvidia's GPU, running appropriate device drivers (Windows or Linux) and has appropriate CUDA toolkit installed. If not then you might want to do these first.  

Additionally the current exercise could be performed on Google Colab (https://colab.research.google.com) which offers ```free GPU``` compute
or on any of the Cloud providers (AWS, GCP etc.) with an appropriate GPU machine""")
    
    
    
    st.subheader("Config")
    st.markdown("""The system config used for this PC are given below. Although you can certainly try the exercise with somewhat lower specs on same data 
I'd recommend decreasing the data size to try on severly under spec machines""")
    st.image("./src/blogs/docs/cpu_info_aws.png", caption='CPU Specs (AWS: ml.g4dn.xlarge)')
    st.image("./src/blogs/docs/gpu_info_aws.png", caption='GPU Specs (AWS: ml.g4dn.xlarge)')

    
    
    st.subheader("Python Libraries")
    st.markdown("""The libraries that were used:  
Cupy : https://cupy.dev  
Numpy : https://numpy.org  
Scikit-Learn : https://scikit-learn.org""")
    
    imports = """import cupy as cp
import numpy as np
from sklearn.datasets import make_regression
"""
    st.code(imports, language='python')
   


    st.subheader("Objective")
    st.markdown("""What we're trying to do here is fit a linear regression model on 500 different datasets with 50 features each and each feature has 1000 data points.In total that is 25 million data points.
However, the goal is that each dataset should end up with its own set of model parameters and we want to this is GPU so we would like to avoid any loops.
It so happens that linear algebra helps us express the problem in a way that these 25 million data points can all be treated together and the model fit with just one line of code following the familiar formulation for regression problems ```Y = X . theta```
""")
    st.markdown("##### Create data")
    
    create_data = """n_datasets=500  # number of different datasets, consider each dataset as an excel table if you will
n_samples = 1000 # number of data points in each dataset i.e. number of rows in each excel table
n_features = 50 # number of features in each dataset i.e. number of columns in each excel table

Y = np.empty((n_samples,)) # Dependent variable (target) 
X = np.empty((n_samples,)) # Independent variables (predictors)

for i in range(n_datasets): # iterate once for each dataset 
    # Create dummy regression data i.e. a table for each dataset
    x,y = make_regression(n_samples=n_samples, 
                          n_features=n_features, 
                          n_informative=n_features)
                  
    # Join all the tables along their columns to create a master table
    Y = np.c_[Y,y] 
    X = np.c_[X,x]

X = X[:,1:] # Since we initialised Independent variables with an empty array drop the first column
Y = Y[:,1:] # Since we initialised dependent variable with an empty array drop the first column

# Up until now the data had been in CPU memory now we can transfer all the data to GPU memory (using Cupy) to leverage GPU computing
Xcd = cp.asarray(X)
Ycd = cp.asarray(Y)

"""
    st.code(create_data, language='python')
    

    st.subheader("Now for the good part")
    st.markdown("""We can fit the model with just one line of code here and although it may look simple and familiar it's good to keep a note of how ```theta``` would be shaped here and how does it ensure we have separate model parameters for each dataset (group)""")
    fit_models = """theta = cp.dot(cp.linalg.pinv(Xcd),Ycd) # This is the closed form solution for linear regression 
theta.shape
(25000,500) """
    st.code(fit_models, language='python')
    
    
    
    st.subheader("Performance")
    st.image('./src/blogs/docs/gpu_model_fit_comparison.png')
    st.markdown("""As can be seen above, the top part displays time taken on ```GPU (~2 sec)``` vs at the bottom time taken on ```CPU (~ 8 sec)```.  
The speed up gains are very real since the CPU implementation is also fully vectorised and uses numpy.  
The data set we've used is big but is by no means huge for a typical data science workflow. Gains can be immense when even larger datasets are used""")
    
    
    
    st.subheader("Notes")
    st.markdown("""1. ```Theta``` is a non square matrix of (n_datasets*n_features) x n_datasets. Each column **j** in theta, we will have zeros (very small values) correspoding to columns of X that do not contribute to the predictions of group **j**  
2. We use pinv and not inv(Xcd.T@Xcd) since memory allocation for Xcd.T@X will be huge so this approach is more efficient  
3. An alternate formulation can be where **X** is designed to be (n_datasets x n_samples x n_features) matrix and treating the problem as solving for tensors

""")
    
    st.markdown(""" ###
    ---
    [Prateek](https://www.linkedin.com/in/prteek/ "LinkedIn")  
    [Repository](https://github.com/prteek/IO/ "Github")

    """)
    