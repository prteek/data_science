# author           : Prateek
# email            : prateekpatel.in@gmail.com
# description      : Drive cycle characterisation for real world fuel economy prediction


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import math
from sklearn.decomposition import PCA
import streamlit as st

def run():
    st.title('Drive cycle characterisation for Fuel Economy analysis')
    st.markdown("""This analysis is based on the SAE technical paper by H. Cho and V. D'urbal.  
[Link to the paper](https://www.sae.org/publications/technical-papers/content/2018-01-0644/)  

The paper aims to establish metrics for real world drive cycles that can be predictors of fuel (energy) consumption.  
Building on those metrics it proposes a methodology to compare drive cycles based on their aggressiveness or city/highway driving characteristics from fuel consumption point of view.  

Central idea is to take the contributors of energy consumption and express them as a function of vehicle speed. Then condense them into independent variables so that based on these variables drive cycles can be compared quantitatively.""")
    
    st.latex(r"""\frac{V_{fuel}}{d} \rho_{fuel} Q_{LHV} \eta_{PT} = \frac{E_{PT}}{d} = \frac{E_{rolling} + E_{aero} + E_{+kinetic}}{d} \hspace{1cm} (1)""")
    
    st.latex(r"""E_{rolling} = \int{F_{rolling}.vdt} = c_{rolling}mg \int{v dt} = c_{rolling} mg.d \hspace{1cm} (2)""")
    
    st.latex(r"""
    E_{aero} = \int{F_{aero}.vdt} = \frac{1}{2} \rho_{air}c_{d}A\int{v^{3} dt} = \frac{1}{2} \rho_{air}c_{d}A \Sigma (v^{3}\Delta t) \hspace{0.5cm} (3)""")
    
    st.latex(r"""E_{+kinetic} = \frac{1}{2}m \Sigma \Delta _{+} (v^{2})             \hspace{1cm} (4) """)

    st.latex(r""" where, \Delta _{+} (v^{2}) = \begin{cases} \Delta (v^{2}), \Delta (v^{2}) >=0 \\    
    0, \Delta (v^{2}) < 0 \\
    \end{cases}
    """)

    st.latex(r"""
    \frac{E_{PT}}{d} = c_{rolling}mg + \frac{1}{2} \rho_{air}c_{d}A \Sigma (v^{3}\Delta t) + \frac{1}{2}m \Sigma \Delta _{+} (v^{2})
 
 """)
    
    st.markdown("""
    **RCS**: Relative cubic speed  
    **PKE**: Positive kinetic energy """)
    
    st.latex(r""" RCS = \frac{\Sigma(v^{3} \Delta t)}{d} \hspace{1cm} PKE = \frac{\Delta _{+}(v^{2})}{d} """)
    
    st.markdown("""Now, based on these 2 metrics the analysis is built to assess the aggressiveness and city/motorway charateristics of drive cycles""")
    
    st.markdown("""### Generate dummy data of RCS and PKE  
The paper uses data from 900 UK based B segment vehicles driving for about 10 days each (so each data point below is one vehicle).  
Since, we don't have that data available to us we generate data based on the description of the data (log(RCS) and PKE) given in the paper (std, cov).  
The generated data appears very similar to the actual data used in the paper. """)



    no_cycles = 1000
    np.random.seed(42)

    mu_pke, std_pke = 0.35, 0.08  # Visually checking histogram in report

    mu_log_rcs, std_log_rcs = 5.5, 0.5  # Visually checking histogram in report

    covariance = -0.4 * std_pke * std_log_rcs  # covariance of log_rcs and pke
    mu = [mu_log_rcs, mu_pke]
    cov = [[std_log_rcs ** 2, covariance], [covariance, std_pke ** 2]]

    log_rcs, pke = np.random.multivariate_normal(mu, cov, no_cycles).T
    rcs = np.exp(log_rcs)

    mu_data = [np.mean(log_rcs), np.mean(pke)]
    cov_data = np.cov([log_rcs, pke])

    fig = go.Figure()
    fig.add_scatter(x=rcs, y=pke, mode='markers', opacity=0.7)
    
    fig.update_layout({"title": "RCS vs PKE",
                      "xaxis_title":"relative cubic speed [m2/s2]",
                      "yaxis_title":"positive kinetic energy [m/s2]"})
    
    st.plotly_chart(fig)

    # In[4]:


    def calculate_rcs_pke(speed_mps, time_step=1):
        cycle_distance = np.trapz(speed_mps, np.arange(0, len(speed_mps), time_step))
        rcs = np.sum(speed_mps ** 3) * time_step / cycle_distance
        pke = (
            np.sum(np.diff(speed_mps ** 2) * (np.diff(speed_mps ** 2) > 0))
            * time_step
            / cycle_distance
        )

        return rcs, pke


    def mah_distance(log_rcs_value, pke_value, mu=mu_data, cov=cov_data):
        rcs_pke_vector = np.array([log_rcs_value, pke_value])
        mah_dist = np.sqrt(
            np.transpose(rcs_pke_vector - np.array(mu))
            @ np.linalg.inv(cov)
            @ (rcs_pke_vector - np.array(mu))
        )
        return mah_dist


    def normquantplot(dataVect, title="data"):
        """ NORMQUANTPLOT plot the normal quantile plot to check if the data is approximately normal """
        dataVect = np.sort(dataVect)
        mm = np.mean(dataVect)
        dev = np.std(dataVect)
        per = (np.arange(len(dataVect)) + 1) / len(dataVect)
        zvals = norm.ppf(per)
        fig = go.Figure()
        fig.add_scatter(x=dataVect, y=zvals, mode='markers', name=title)
        fig.add_scatter(x=[mm - 2 * dev, mm + 2 * dev], 
                        y=[-2, 2], mode='lines', line={'dash':'dash'},
                        name="Approximate Normal line")
        
        fig.update_layout({"title": f"Normal Quantile plot of {title}",
                      "xaxis_title":title,
                      "yaxis_title":"Normal Score"})
        
        st.plotly_chart(fig)
        
        return None

        
    def ECDF(x):
        x = np.sort(x)
        n = len(x)
        def _ecdf(v):
            # side='right' because we want Pr(x <= v)
            return (np.searchsorted(x, v, side='right') + 1) / n
        return _ecdf

    st.markdown("""### RCS and PKE of regulation cycles  
For the **cycles**, their RCS and PKE can be calculated from speed data in similar manner but since the data is already available in the paper, we use that """)


    cycles_data = np.array(
        [
            [515, 0.308],
            [362, 0.223],
            [253, 0.346],
            [511, 0.140],
            [761, 0.418],
            [230, 0.410],
            [377, 0.451],
            [191, 0.356],
        ]
    )
    cycle_names = ["wltp", "nedc", "ftp75", "hwfet", "us06", "sc03", "la92", "jc08"]

    fig = plt.figure()
    fig = go.Figure()
    fig.add_scatter(x=rcs,y=pke, mode='markers', opacity=0.7, name="data")
    fig.add_scatter(x=cycles_data[:, 0], y=cycles_data[:, 1], 
                    mode='markers+text', marker={'size':10}, 
                    text=cycle_names, textposition="bottom center",
                    name="cycles")
    
    fig.update_layout({"title": "RCS vs PKE",
                      "xaxis_title":"relative cubic speed [m2/s2]",
                      "yaxis_title":"positive kinetic energy [m/s2]"})
        
    st.plotly_chart(fig)

    st.markdown("""Since log(RCS) and PKE are both approximately normally distributed, RCS is replaced with log(RCS) for further analysis.  
Now a bivariate normal (chi2) distribution can be fit to the data.   
""")

    fig = go.Figure()
    fig.add_scatter(x=log_rcs, y=pke, mode='markers', opacity=0.5, name="data")
    fig.add_scatter(x=np.log(cycles_data[:, 0]), y=cycles_data[:, 1], 
                    mode='markers+text', marker={'size':10}, 
                    text=cycle_names, textposition="bottom center",
                    name="cycles")

    fig.update_layout({"title": "LogRCS vs PKE",
                      "xaxis_title":"log relative cubic speed",
                      "yaxis_title":"positive kinetic energy [m/s2]"})
        
    st.plotly_chart(fig)
    
    st.markdown("""But since there is slight correlation between these parameters, a good check of normality is quantile plot of Mahalanobis distance of data points from the mean of the data.
""")
    
    d = [mah_distance(lrcs_i, pke_i, mu=mu_data, cov=cov_data)
        for lrcs_i, pke_i in zip(log_rcs, pke)]

    normquantplot(d, "Mahalonobis distance")
    

    st.markdown("""### Principal Component analysis  
To remove the correlation between parameters, we can take the principal components in the data and use them as our primary axes of visualisation. This is done in 3 steps: 

1. Find the principal components, and bring the original axes to the mean of the data  
2. Rotate the original axes to align with the Principal components  
3. Normalise the data points by eigen values to get comparable x and y axes  
""")
    
    u, l, v = np.linalg.svd(cov_data)
    theta_svd   = math.atan(u[0][1]/u[0][0])

    s       = np.sqrt(np.identity(2)*l)
    r       = np.array([[np.cos(theta_svd), -np.sin(theta_svd)], [np.sin(theta_svd), np.cos(theta_svd)]])

    def transform_to_pc(log_rcs_value, pke_value):
        rcs_pke_vector = np.array([log_rcs_value, pke_value])
        transformed    = np.linalg.inv(s)@np.linalg.inv(r)@(rcs_pke_vector - np.array(mu_data))
        return transformed

    transformed_data   = np.array([transform_to_pc(lrcs_i, pke_i) for lrcs_i, pke_i in zip(log_rcs, pke)])
    
    fig = make_subplots(rows=1, cols=2, subplot_titles = ('PCs of data', 
                                        'Data aligned to PCs'))
    fig.add_trace(
        go.Scatter(x=log_rcs, y=pke, mode='markers', opacity=0.7, name='data'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=[mu_data[0], u[0][0]], y=[mu_data[1], u[0][1]], 
                   mode='lines', line={'dash':'dash'}, name='PC2'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=[mu_data[0], u[1][0]], y=[mu_data[1], u[1][1]], 
                   mode='lines', line={'dash':'dash'}, name='PC1'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=transformed_data[:,0], y=transformed_data[:,1], 
                   mode='markers', name='transformed data'),
        row=1, col=2
    )

    fig.update_layout(height=450, width=800, title_text="Principle Components decomposition")
    fig.update_xaxes(title_text="log rcs", range=[0, 8], row=1, col=1)
    fig.update_yaxes(title_text="pke [m/s2]", range=[0,0.8], row=1, col=1)
    fig.update_xaxes(title_text="Principal component 1", range=[-4, 4], row=1, col=2)
    fig.update_yaxes(title_text="Principal component 2", range=[-4, 4], row=1, col=2)
    
    st.plotly_chart(fig)

    u, l, v = np.linalg.svd(cov_data)
    theta_svd = math.atan(u[0][1] / u[0][0])

    s = np.sqrt(np.identity(2) * l)
    r = np.array(
        [[np.cos(theta_svd), -np.sin(theta_svd)], [np.sin(theta_svd), np.cos(theta_svd)]]
    )


    def transform_to_pc(log_rcs_value, pke_value):
        rcs_pke_vector = np.array([log_rcs_value, pke_value])
        transformed = (
            np.linalg.inv(s) @ np.linalg.inv(r) @ (rcs_pke_vector - np.array(mu_data))
        )
        return transformed


    # In[8]:


    transformed_data = np.array(
        [transform_to_pc(lrcs_i, pke_i) for lrcs_i, pke_i in zip(log_rcs, pke)]
    )

    transformed_cycles = np.array(
        [transform_to_pc(np.log(rcs_i), pke_i) for rcs_i, pke_i in cycles_data]
    )


    st.markdown("""In a similar way the cycles and trip data can be transformed.  
    
Now, since we have 2 independent parameters for each cycle, we can use chi2 distribution which is a function of distance from the mean (or in our case distance from 0).  

The data points can be normalised by total probability of data to visualise on a radar chart.  

The new coordinate system from the PCA does not necessarily have physical properties directly related to the original parameter set. SI and DI represent the normalised relative distance from a nominal driver with average speed and moderate driving style. CI indicates the distance from the nominal driver in any direction and does not represent fuel consumption.
""")

    # ### Alternate approach to Axes transformations

    # In[9]:


    pca = PCA(n_components=2)
    X = np.c_[log_rcs, pke]

    transformed_data_pca = pca.fit_transform(X)
    transformed_data = transformed_data_pca / np.std(transformed_data_pca, axis=0)

    transformed_cycles_pca = pca.transform(
        np.c_[np.log(cycles_data[:, 0]), cycles_data[:, 1]]
    )
    transformed_cycles = transformed_cycles_pca / np.std(transformed_data_pca, axis=0)

    # distances from mean of the data
    distance = np.sqrt(transformed_data[:, 0] ** 2 + transformed_data[:, 1] ** 2)
    cycle_distance = np.sqrt(transformed_cycles[:, 0] ** 2 + transformed_cycles[:, 1] ** 2)

    mean_transformed = np.mean(transformed_data, axis=0)  # should be [0,0]
    cov_transformed = np.cov(
        transformed_data[:, 0], transformed_data[:, 1]
    )  # Variance should be unchaged but cov=0

    ecdf = ECDF(distance)  # Chi2 distribution

    # data
    ci = ecdf(distance)
    theta = np.array([(math.atan2(y_i, x_i)) for x_i, y_i in transformed_data])

    # cycles
    ci_cycles = ecdf(cycle_distance)
    si = ci_cycles * np.array(
        [math.cos(math.atan2(y_i, x_i)) for x_i, y_i in transformed_cycles]
    )
    di = ci_cycles * np.array(
        [math.sin(math.atan2(y_i, x_i)) for x_i, y_i in transformed_cycles]
    )
    theta_cycles = np.array([(math.atan2(y_i, x_i)) for x_i, y_i in transformed_cycles])

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=ci,
        theta=theta,
        thetaunit = "radians",
        mode='markers',
        name='data',
    ))

    fig.add_trace(go.Scatterpolar(
        r=ci_cycles,
        theta=theta_cycles,
        thetaunit = "radians",
        mode='markers+text',
        text=cycle_names,
        textposition="top center",
        textfont=dict(
            size=12,
            color="Black"
        ),
        marker=dict(size=10, color="red"),
        name='cycles',
    ))

    # Update axis ticks labels to ["Motorway", "Aggressive", "City", "Mild"]
    fig.update_layout(
        polar=dict(
            angularaxis=dict(
                tickmode = 'array',
                tickvals = np.array([0, np.pi/2, np.pi, 3*np.pi/2])*180/np.pi,
                ticktext = ["Motorway", "Aggressive", "City", "Mild"],
                rotation=0,
                direction="counterclockwise",
            ),
        ),
    )

    st.plotly_chart(fig)
    
    st.markdown(""" ###
    ---
    [Prateek](https://www.linkedin.com/in/prteek/ "LinkedIn")  
    [Repository](https://github.com/prteek/IO/ "Github")

    """)

