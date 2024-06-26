import streamlit as st
from src.projects import drive_cycle_characterisation as dcc
from src.projects import strava_project as sp
from src.projects import public_transport_range as ptr
from src.projects import euro_2024_modelling as e2m
from src.projects import imdb_movie_length as iml


class About:
    def __init__(self):
        return None
    
    def run(self):
        st.caption("""Hi ! I am Prateek and this is a collection of my side projects.  
There is a slim chance you may enjoy these too.  
If you'd like to get in touch, please use the details below:

        """)
        
        st.markdown("""
---  
[Mail](mailto:prteek@icloud.com "Mail")  
[Github](https://github.com/prteek/ "Github")  

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Z8Z243K2E)    
""")


projects = {
    'Euro 2024 model': e2m,
    # "Strava": sp,
    "Drive cycle characterisation": dcc,
    "Movies getting longer ?": iml,
    "Public transport range": ptr,
    "About": About(),
}

project_titles = list(projects.keys())
tabs = st.tabs(project_titles)

for i, tab in enumerate(tabs):
    with tab:
        projects[project_titles[i]].run()
