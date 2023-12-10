# author            : Prateek
# email             : prateekpatel.in@gmail.com
# description       : simulation to determine number of samples required for a survival analysis study

import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from scipy.special import kl_div
import time as TT
from scipy.interpolate import interp1d
import streamlit as st
from plotly import graph_objects as go

def run():
    
    st.title("Estimating number of samples for Survival Analysis study")
    st.markdown("""Idea here is to estimate the minimum number of samples that would be required to conduct a specific modelling of battery ageing.   
    Since, classical statistical approach seem too convoluted, I opted to make a resampling simulation to make the analysis easy to understand and control.  
    """)

    np.random.seed(42)  # For repeatability

    st.markdown("""### Short overview of methodology:  
1. Synthetically generate battery lifetimes for the entire fleet, based on assumptions about fleet size and nature of how batteries have failed in the past  
2. Randomly select a small number of batteries (trucks) from the fleet batteries to observe for the logging duration  
3. Compare the lifetimes of selected sample of batteries (trucks) to the fleet, and how close/further they are from fleet lifetimes  
4. If they are close enough and if the process is repeated again and again, and this closeness in lifetimes doesn't change significantly, then the sample size chosen can be acceptable  
    ###
    """)


    st.markdown(""" ### Assumptions:
    ####
    """)

    fleet_size = st.slider("Fleet size (# of trucks)", 50, 500, 100, step=50)  # assumption


    """
    #### 1 month = 4 weeks
    """

    battery_perfectly_healthy_until_month = st.slider(
        "No batteries fail until [months]", 0, 12, 2, step=1
    )  # 2 months they are fine
    battery_perfectly_healthy_until_week = battery_perfectly_healthy_until_month * 4

    mean_battery_age_months = st.slider(
        "Mean battery life [months] ",
        battery_perfectly_healthy_until_month,
        12,
        max(5, battery_perfectly_healthy_until_month),
    )  # 5 months * 4 weeks
    mean_battery_age_weeks = mean_battery_age_months * 4


    # Generate battery life data. This should be close representation of reality and our target for modelling

    fleet_age_distribution = (
        np.random.exponential(
            mean_battery_age_weeks - battery_perfectly_healthy_until_week, fleet_size
        )
        + battery_perfectly_healthy_until_week
    )

    st.markdown("""
    #### For any analysis down the line, it is important to understand first that, based on our selection are we even capturing the behaviour that we want to model ?

    #### The natural behaviour of battery failures itself is first source of uncertainity. The lifetimes of batteries in the fleet has a wide distribution.

    #### This is assumed to be ```exponential``` as understood from failure patterns of machine components. 

    """)

    fig = go.Figure(data=[go.Histogram(x=fleet_age_distribution)])
    
    fig.update_layout({'title':'Distribution of battery life for fleet', 
                       'xaxis_title':'Battery life [weeks]', 
                       'yaxis_title': 'Number of Trucks'})
    
    st.plotly_chart(fig)

    # Choose trucks and model their age

    ### Setting data collection experiment parameters
    size_options = np.arange(0, fleet_size + 1, 5)[1:]  # Ignore case with 0 trucks
    no_censored = (
        False
    )  # Set battery failure events to not be right-censored owing to short logging durations


    st.markdown("""### Studying Battery failures""")

    logging_duration_months = st.slider("Logging duration", 0, 12, 6, step=1)
    logging_duration_weeks = logging_duration_months * 4

    st.markdown(f"""
    During the logging duration ({logging_duration_weeks/4} months) trucks are chosen and each one of their batteries may or may-not survive past this logging duration.

    Ideally the more failures are captured during logging, the better.

    The logging duration thus becomes important in modelling accuracy and its effect can be seen later in AUC (area under the curve) plot also.

    ###
    """)


    # Initialising plot and result variables
    auc = []
    norm_auc = []
    kldvg = []
    fig = go.Figure()
    kmf = KaplanMeierFitter()
    # Full fleet data model
    observed_event = fleet_age_distribution <= 9999
    time_fleet = np.arange(int(max(fleet_age_distribution)) + 1)
    kmf.fit(fleet_age_distribution, observed_event, timeline=time_fleet)
    survival_prob_fleet = np.array(kmf.survival_function_.KM_estimate)
    prob_lookup = interp1d(time_fleet, survival_prob_fleet, kind="nearest")

    button_input = st.button("Randomise and re-run")
    if button_input:
        np.random.seed(int(TT.time()))

    n_iters = 10
    auc = np.zeros((len(size_options), n_iters))
    norm_auc = np.zeros((len(size_options), n_iters))
    kldvg = np.zeros((len(size_options), n_iters))
    tic = TT.time()

    my_bar = st.progress(0)
    for i_iter in range(n_iters):
        my_bar.progress((i_iter * 100 + 100) // n_iters)

        for i, n_trucks in enumerate(size_options):
            n_trucks = int(n_trucks)
            # assumption is monitoring starts on any age of batteries, likely even to fail in the first week of logging. This gives us more chance to see failure in logged data
            trucks_age_weeks = np.random.choice(
                fleet_age_distribution - battery_perfectly_healthy_until_week,
                n_trucks,
                replace=False,
            )

            if no_censored:
                observed_duration_weeks = 9999
            else:
                observed_duration_weeks = logging_duration_weeks

            kmf = KaplanMeierFitter()
            observed_event = trucks_age_weeks <= observed_duration_weeks

            time = np.arange(logging_duration_weeks + 1)
            kmf.fit(trucks_age_weeks, observed_event, timeline=time)
            survival_prob = np.array(kmf.survival_function_.KM_estimate)

            time = time + battery_perfectly_healthy_until_week
            logging_index = np.where(time <= logging_duration_weeks)

            survival_prob_fleet_matching = prob_lookup(time[logging_index])

            kldvg_i = kl_div(survival_prob_fleet_matching, survival_prob[logging_index])

            norm_auc[i, i_iter] = (
                np.trapz(
                    survival_prob[logging_index] * (1 - 10 * kldvg_i), time[logging_index]
                )
                + battery_perfectly_healthy_until_week
            )

            auc[i, i_iter] = (
                np.trapz(survival_prob[logging_index], time[logging_index])
                + battery_perfectly_healthy_until_week
            )

            kldvg[i, i_iter] = np.nanmean(kldvg_i)

            #  Plot if number of trucks is very low or very high (ROI) and only 1 iteration
            if i_iter == 0:
                if i > 3 and i <= len(size_options) - 3:
                    continue
                    
                fig.add_trace(go.Scatter(x=time[logging_index], 
                                         y=survival_prob[logging_index],
                                        name=str(n_trucks)
                                         + " trucks : "
                                         + str(sum(observed_event))
                                         + " failed",
                                        line={'shape':'hv'},
                                        mode='lines'
                                        )
                             )


    fig.add_trace(go.Scatter(x=time_fleet, 
                             y=survival_prob_fleet, 
                             name="reality",
                            line={'shape':'hv'}))
    fig.add_scatter(x=[logging_duration_weeks, logging_duration_weeks],
                   y = [0,1], name="logging duration", mode='lines', line={'dash':'dot'})

    fig.update_layout({"title": "Survival Probabilities",
                      "xaxis_title": "time [weeks]",
                      "yaxis_title": "est. probability of survival"})
    
    st.plotly_chart(fig)
    
    auc_org = auc
    norm_auc_org = norm_auc

    auc = np.mean(auc_org, axis=1)
    std_auc = np.std(auc_org, axis=1)
    norm_auc = np.mean(norm_auc_org, axis=1)
    kldvg = np.mean(kldvg, axis=1)
    fleet_auc = np.trapz(survival_prob_fleet, time_fleet)


    st.markdown(f"""
    #### The plot below represents that for various selection of number of trucks to be studied, how close the observed failures can get to:  

    * Best set of observations that can be possible with {logging_duration_months} months of logging
    * Reality i.e. if entire fleet were studied until all batteries failed

    AUC (Area under the Survival Curve) for each curves above can be considered an average life probability (weeks) by the observations made by selecting corresponding number of trucks/batteries.

    The black lines envelope an area in which the AUC will lie for a given number of trucks selection if the experiment is repeated multiple times. (+/- 2*std = 95 percent bounds)

    It can be seen that for less number of samples in the study the AUC can vary quite significantly and we have high uncertainity in observing true nature of the fleet. 

    As we add more samples to our study the predictions become more regular, even if experiment is repeated multiple times, this mitigates the effect of randomness and we can be more confident that the observed behaviour is not a mere result of ```chance```.

    This converging sort of behaviour may be considered analogous to convergence behaviour of PID, and as can be seen a good selection of number of samples happens around 30 samples (thumb rule from statistics).

    #### So our target number of samples (ideally) should be:
    * A number about which AUC uncertainity is less
    * The observed average life probability (AUC) is within +/- 1 week of best possible AUC (best case is when all the members of fleet are included in the study)


    Also, notice that the logging duration brings *best model* closer to *reality*.
    """)


    fig = go.Figure()
    
    fig.add_scatter(x=size_options, 
                   y=auc // 1,
                   name=f"mean of AUC over {n_iters} iterations")
    fig.add_scatter(x=size_options, y=(auc + 2 * std_auc) // 1, 
                    name="mean + 2*std", line={'dash':'dash'})
    fig.add_scatter(x=size_options, y=(auc - 2 * std_auc) // 1, 
                    name="mean - 2*std", line={'dash':'dash'})
    
    fig.add_scatter(x=[0, size_options[-1]],y=[auc[-1] // 1, auc[-1] // 1],
                    name="AUC by selecting entire fleet", line={'dash':'dot'}, mode='lines')
    fig.add_scatter(x=[0, size_options[-1]], y=[fleet_auc // 1, fleet_auc // 1],
                    name="reality", line={'dash':'dot'}, mode='lines')
    
    fig.update_layout({"title": f"AUC of survival curves, by repeating the exercise {n_iters} times",
                      "xaxis_title": f"number of trucks studied over {logging_duration_weeks / 4} months",
                      "yaxis_title": "AUC (Area under the curve) [weeks]"})
    
    st.plotly_chart(fig)

    elapsed = TT.time() - tic
    print(elapsed)

    st.markdown(""" ###
    ---
    [Prateek](https://www.linkedin.com/in/prteek/ "LinkedIn")  
    [Repository](https://github.com/prteek/IO/ "Github")

    """)