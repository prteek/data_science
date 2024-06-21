import streamlit as st
from src.projects import projects_home as ph
from src.blogs import blog_home as bh




class home_page():
    def __init__(self):
        return None
    
    def run(self):
        st.title('Prateek')
        st.write("""Hi ! I am your just another friendly neighbourhood Data-Scientist.  
Similar to your other friendly neighbourhood costumed crusaders, I believe that "with great power comes great responsibility".  
So I rely on sound logic and principles of statistics, physics and math for designing and analysing stuff that matter.  
In the past I have worked with companies like:
* Suzuki
* Jaguar Land Rover
* Ford
* Allison Transmission
* Cloud Cycle 
* Bricklane

On problems ranging from track testing, Hybrid electric powertrain design, vehicle monitoring system, concrete quality degradation
and real estate behaviour modelling.

I have deep interest in Statistics, Machine learning and Python and I occasionally dabble in Astrophysics and Photography.  

If you'd like to get in touch, please use the details below:

        """)
        
        st.markdown(""" ###  
---  
[Prateek](https://www.linkedin.com/in/prteek/ "LinkedIn")  
[Repository](https://github.com/prteek/IO/ "Github")  

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Z8Z243K2E)    
""")


home = home_page()

projects = {'Home':home,
            'Projects':ph,
            # 'Blog':bh,
           }

projects_tab, about_tab = st.tabs(["Projects", "About"])
with projects_tab:
    projects["Projects"].run()

with about_tab:
    projects["Home"].run()