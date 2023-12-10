import streamlit as st
from src.blogs import chittr as chit
from src.blogs import fit500 as fit500
from src.blogs import problem_solving_with_gpu as pswg
from src.blogs import flickr_analysis as fa
from src.blogs import coronavirus_eda_and_prediction as ceap


def run():
    
    projects = {'Home':chit,
               'Fit 500 models on GPU':fit500,
               # 'Problem solving with GPU':pswg,
               'COVID19 EDA and early stage model':ceap,
               'Flickr analysis':fa}

    page = st.sidebar.selectbox(label='Blogs', options=list(projects.keys()))
    projects[page].run()
    