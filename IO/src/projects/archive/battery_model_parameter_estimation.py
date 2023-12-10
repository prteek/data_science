# author           : Prateek
# email            : prateekpatel.in@gmail.com
# description      : Demonstrates parameter estimation for physical models of Lithiom Ion battery from test data using non-linear regression, and good and bad side of using Decision tree for the same

import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.tree import export_graphviz
import pydotplus
import streamlit as st
from plotly import graph_objects as go

def run():
    st.title('Battery model parameter estimation using non-linear regression')
    st.markdown("""This is the demonstration of an approach where non-linear regression is used to estimate parameters of battery models from test data.  
    A general approach is running a time-series simulation of the model and iterating on the parameters until a good fit is found.
    The general approach is very combursome to code and maintain and also takes longer time to estimate the parameters.  
      
As an exercise in Machine learning Decision tree is also used to avoid physical modelling but it results in incomplete model of the battery.
""")
    def model(X, r0, r1, c1, r2, c2):
        """Dual polarisation model of battery"""
        t = X[:, 0]
        i = X[:, 1]

        voltage = (
            ocv
            - i * r0
            - i * r1 * (1 - np.exp(-t / (r1 * c1)))
            - i * r2 * (1 - np.exp(-t / (r2 * c2)))
        )
        return voltage


    def model2(X, r0, r1, c1):
        """R-RC model of battery"""
        t = X[:, 0]
        i = X[:, 1]
        voltage = ocv - i * r0 - i * r1 * (1 - np.exp(-t / (r1 * c1)))
        return voltage


    # Create some dummy data to test models

    time = np.arange(200)
    I = 150 * np.ones((200, 1)) + np.random.random((200,)) * 10

    X_in = np.c_[time, I]

    # Arbitrary parameters to generate dummy data
    R0 = 5e-3
    R1 = 5e-4
    C1 = 5e4
    R2 = 5e-5
    C2 = 5e4

    ocv = 4.2
    voltage = model(X_in, R0, R1, C1, R2, C2) + np.random.random((200,)) * 0.01 / 2

    st.markdown(""" ### Generating some dummy battery test data
    This data represents a test where battery was discharging a constant current and Voltage response was observed with time.

    """)
    fig = go.Figure()
    fig.add_scatter(x=time, y=voltage, mode='markers')
    fig.update_layout({'title':"Dummy data",
                       'xaxis_title': 'time [s]',
                       'yaxis_title':'cell_voltage [V]'})
    st.plotly_chart(fig)


    # Fitting Dual polarisation model to dummy data
    st.markdown(""" ### Fitting a dual polarisation model with non-linear regression

    """)


    ini = np.array([R0, R1, C1, R2, C2]) * 0.5

    popt, pcov = curve_fit(model, X_in, voltage, p0=ini, bounds=(0, np.Inf))

    fig = go.Figure()
    fig.add_scatter(x=time, y=voltage, mode='markers', name="data")
    fig.add_scatter(x=time, y=model(X_in, *popt), mode="lines", name="DP model fit to data")
    fig.update_layout({'xaxis_title': 'time [s]',
                       'yaxis_title':'cell_voltage [V]'})
    st.plotly_chart(fig)

    df = pd.DataFrame(
        np.c_[[R0, R1, C1, R2, C2], popt],
        ["R0", "R1", "C1", "R2", "C2"],
        columns=["Actual value", "Fitted value"],
    )

    st.markdown("""Parameters of fitted Dual polaristion model""")
    df


    # fitting R-RC model to dummy data
    st.markdown(""" ### Fitting a single polarisation model with non-linear regression

    """)
    ini = np.array([R0, R1, C1]) * 2

    popt, pcov = curve_fit(model2, X_in, voltage, p0=ini, bounds=(0, np.Inf))
    fig = go.Figure()
    fig.add_scatter(x=time, y=voltage, mode='markers', name="data")
    fig.add_scatter(x=time, y=model2(X_in, *popt), mode="lines", name="R-RC model fit to data")
    fig.update_layout({'xaxis_title': 'time [s]',
                       'yaxis_title':'cell_voltage [V]'})
    st.plotly_chart(fig)
    

    df = pd.DataFrame(
        np.c_[[R0, R1, C1], popt],
        ["R0", "R1", "C1"],
        columns=["Actual value", "Fitted value"],
    )

    st.markdown("""Parameters of fitted Dual polaristion model""")
    df


    # In[5]:


    # Working with real data

    st.markdown("""### Let's check how this non linear regression fitting performs on real-world data

    """)

    df = pd.read_csv("./src/projects/docs/data.txt", delimiter="\t")
    df.columns = ["time", "current", "voltage"]

    time = df["time"] - df["time"][0]
    current = -df["current"]
    voltage = df["voltage"]

    ocv = 3.75

    X_data = np.c_[time, current]

    # Ballpark estimates
    ini = np.array([0.0005, 0.0001, 10000, 0.0001, 100000])

    popt, pcov = curve_fit(model, X_data, voltage, p0=ini, bounds=(0, np.Inf))

    print(
        pd.DataFrame(
            np.c_[popt.T, np.diag(pcov).T],
            ["R0", "R1", "C1", "R2", "C2"],
            columns=["Fitted parameters", "Variance of fit"],
        )
    )
    
    
    fig = go.Figure()
    fig.add_scatter(x=time, y=voltage, mode='markers', name="logged voltage")
    fig.add_scatter(x=time, y=model(X_data, *popt), mode="lines", name="modelled voltage")
    fig.update_layout({'xaxis_title': 'time [s]',
                       'yaxis_title':'cell_voltage [V]'})
    st.plotly_chart(fig)

    error = voltage - model(X_data, *popt)
    msg = "\n RMSE: %f" % (np.sqrt(np.mean(error ** 2)))

    st.write(msg)
    print(msg)


    # In[6]:


    st.markdown("""### Demonstration of failure of machine learning, to model battery behaviour
    """)
    # Fit the model to logged data

    X_train = X_data

    X = np.array(X_train)
    y = np.array(voltage)

    tree = DecisionTreeRegressor()

    tree.fit(X, y)



    st.markdown("""#### Run the model on a synthetic data which has a very long current pulse

    """)

    time = np.arange(200)
    I = 150 * np.ones((200, 1)) + np.random.random((200, 1)) * 10

    X_test = np.c_[time, I]

    v_predict = tree.predict(X_test)
    v_predict_rf = tree.predict(X_test)

    fig = go.Figure()
    fig.add_scatter(x=X_train[:, 0], y=voltage, mode='markers', name="training data")
    fig.add_scatter(x=X_test[:, 0], y=v_predict, mode='markers', name="test data")
    fig.update_layout({'xaxis_title': 'time [s]',
                       'yaxis_title':'cell_voltage [V]'})
    
    st.plotly_chart(fig)

    st.markdown(""" What this shows is that to get a machine model that represents holistic picture of battery behaviour, across current draw, duration of pulse, SOC, temperature:

    The model needs to be trained on several combinations of these parameters which will be too data-intensive exercise

    """)
    # In[8]:


    dot_data = export_graphviz(
        tree,
        out_file=None,
        feature_names=["time", "current"],
        class_names=["voltage"],
        rounded=True,
        filled=True,
    )

    graph = pydotplus.graph_from_dot_data(dot_data)

    # graph.write_png("docs/visualise_decision_tree.png")

    plt.close('all')

    st.markdown(""" ###
    ---
    [Prateek](https://www.linkedin.com/in/prteek/ "LinkedIn")  
    [Repository](https://github.com/prteek/IO/ "Github")

    """)
