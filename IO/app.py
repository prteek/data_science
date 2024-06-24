import streamlit as st
from src.projects import drive_cycle_characterisation as dcc
from src.projects import strava_project as sp
from src.projects import public_transport_range as ptr
from src.projects import euro_2024_modelling as e2m


class About:
    def __init__(self):
        return None
    
    def run(self):
        st.title('Prateek')
        st.write("""Hi ! I am Prateek and this app is a collection of short projects that I've made in my own time.  
There is a slim chance you may also enjoy these too.  
If you'd like to get in touch, please use the details below:

        """)
        
        st.markdown(""" ###  
---  
[Mail](mailto:prteek@icloud.com "Mail")  
[Repository](https://github.com/prteek/IO/ "Github")  

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Z8Z243K2E)    
""")


projects = {
    'Euro 2024 model': e2m,
    "Strava": sp,
    "Drive cycle characterisation": dcc,
    "Public transport range": ptr,
    "About": About(),
}

project_titles = list(projects.keys())
tabs = st.tabs(project_titles)

for i, tab in enumerate(project_titles):
    with tab:
        projects[project_titles[i]].run()
