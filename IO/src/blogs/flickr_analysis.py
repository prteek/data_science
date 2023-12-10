import pandas as pd
import flickr_api
import os
import json
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer as FT
from functools import partial
from dotenv import load_dotenv
load_dotenv('local_credentials.env')
import logging
import streamlit as st
import altair as alt
import numpy as np


logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s %(funcName)s : %(message)s"
)


def save_json_as_file(photos_json:json, filename:str) -> str:
    """Save photos exported as json to file. This increases reusability"""
    with open(filename, 'w') as f:
        logging.info(f'Writing {filename}')
        f.write(photos_json)
        
    return filename
    

def get_exif_data_dict(photo:flickr_api.objects.Photo)-> dict:
    exif_data = photo.getExif()
    exif_dict = dict()
    for data in exif_data:
        tag = data.tag
        raw = data.raw
        exif_dict[tag] = raw
        
    return exif_dict
    
    
    
def export_photos_as_json(photos:flickr_api.objects.FlickrList) -> json:
    """Export Exif data in list of photo objects as json against their photo id"""
    photos_json = dict()
    for photo in photos:
        logging.info(f"Processing {photo.title}")
        photos_json[photo.id] = get_exif_data_dict(photo)
            
    return json.dumps(photos_json)


def get_user_photos(username:str) -> flickr_api.objects.FlickrList:
    """Get all photos from flicker for a given user"""
    flickr_api.set_keys(api_key=os.environ['FLICKR_API_KEY'], 
                    api_secret=os.environ['FLICKR_API_SECRET'])

    user = flickr_api.Person.findByUserName(username)
    photos = user.getPhotos()
    return photos


download_user_photos_exif_pipeline = make_pipeline(FT(get_user_photos), 
                                    FT(export_photos_as_json), 
                                    FT(partial(save_json_as_file, filename='photos.json')),
                                    )


def intro():
    intro_text = """I’ve been using Flickr for inspiration and came across june1777’s work through a blog. His photos capture scenes where there is roughly equal distribution between light and dark, and doing so in a way that is often striking and beautiful. He’s a fan of fast prime lenses shot wide open, often wandering the streets of South Korea after dusk, letting the juxtaposition of the glow from the evening sea of lights settle in against the urban landscape.

Similar to the blog I discovered june1777 on I thought it would be a good exercise to attempt to grab whatever exif data I could via the flickr API and analyze june1777’s photo data."""
    
    return intro_text



def run():
    
    st.title("Flickr analysis")
    
    st.markdown(intro())
        
    # get photos for user
    user = 'june1777'
    filename = 'src/projects/docs/june1777.json'
    download_user_photos_exif_pipeline.set_params(**{'functiontransformer-3__kw_args':
                                                     {'filename':filename}})

    if os.path.exists(filename): # file exists do nothing
        pass 
    else: # Else download user data as filename
        filename = download_user_photos_exif_pipeline.transform('june1777')

    # load data as dataframe
    photos_df = pd.read_json(filename, orient='index')

    st.markdown("""### Let's see what kind of camera models were used""")
    fig = (alt
           .Chart(photos_df
                  .value_counts(subset=['Model'], normalize=True)
                  .to_frame(name='Fraction')
                  .reset_index(), title='Camera models')
           .mark_bar()
           .encode(x='Model:N',y='Fraction:Q', tooltip='Fraction:Q')
           .interactive()
          )
           
    st.altair_chart(fig, use_container_width=True)
    st.markdown("""32% with a Canon EOS 5D  
24% with a Contax N Digital  
14% with a Sony A7 MK2  
He also used Nex-3 for a bit  
Also worth noting about 8% of his photos were film scans.
    """)
    
    st.markdown("#")
    st.markdown("""### Let's take only digital photos and drop the ones that are from Fujifilm scanner 'SP-3000' which won't have any exif data""")
    
    scanners = ['SP-3000','QSS']
    digi_photos_df = photos_df.loc[-photos_df['Model'].isin(scanners)]
    
    st.markdown(f"""Checking the shape of digital photots DataFrame:{digi_photos_df.shape}.  
The number of columns is way high for us to make any sense of it. Let’s see if we can filter columns with at least 70% non empty values""")

    digi_photos_df = digi_photos_df.loc[:, digi_photos_df.notnull().mean().sort_values(ascending=True) > 0.7]
    
    st.write(f"Number of columns after filtering: {digi_photos_df.shape[1]}")
    st.markdown("""### Let's also check the ```Make``` of the models""")
    st.table(digi_photos_df.value_counts(subset=['Make', 'Model']).to_frame(name='Count').reset_index())

    st.markdown("#")
    st.markdown("""### Which ISO values are most common ?""")
    
    col1,col2 = st.columns((1,2))
    fig = (alt
           .Chart(digi_photos_df.value_counts(subset=['ISO']).to_frame(name='Counts').reset_index(),
                 title='ISO Distribution')
           .mark_bar()
           .encode(x='ISO:Q', y='Counts:Q', tooltip='Counts:Q')
           .interactive()
          )
    
    col1.altair_chart(fig, use_container_width=True)

    st.markdown(""" ISO values are logarithmic in nature so this doesn't give us much insight directly, so we need to take log of these values """)
    
    log_df = (digi_photos_df
              .assign(LogISO=np.log(digi_photos_df['ISO']))
              .value_counts(subset=['LogISO'])
              .to_frame(name='Counts')
              .reset_index()
             )
    
    log_df['ISO'] = np.exp(log_df['LogISO'])
                    
    fig = (alt
           .Chart(log_df, title='LogISO Distribution')
           .mark_bar()
           .encode(x='LogISO', 
                   y='Counts:Q', tooltip=alt.Tooltip(['Counts:Q', 'ISO:Q']))
           .interactive()
          )
    
    col2.altair_chart(fig, use_container_width=True)
    
    st.markdown("""Using 2nd chart it is clear that 1600 is the most common ISO setting which makes sense since these photos are shot in the evening hours""")
    
    fig = (alt
           .Chart(digi_photos_df
                 .assign(iso_max = lambda x: pd.cut(x['ISO'], bins=[0,400,800,1200,1600,2000, 50000], 
                                          labels=[400, 800,1200,1600,2000, 50000]),
                        aperture_max = lambda x: pd.cut(x['MaxApertureValue'], bins=[0,2,4,6,20], 
                                          labels=[2,4,6,20]))
                 .groupby(['iso_max', 'aperture_max'])
                 .agg('count')
                 .reset_index()
                 .rename(columns={'ISO':'count'})
                 )
           .mark_rect()
           .encode(x='iso_max:O', y='aperture_max:O', color='count:Q', tooltip=['count:Q', 'iso_max:Q', 'aperture_max:Q'])
           .interactive()
          )
    
#     fig = (alt
#            .Chart(digi_photos_df
#                      .groupby(['ISO', 'MaxApertureValue'])
#                      .agg('count')
#                      .reset_index()
#                      .rename(columns={'Make':'count'})
#                      )
#            .mark_rect()
#            .encode(x='ISO:O', y='MaxApertureValue:O', color='count:Q', tooltip=['count:Q', 'ISO:Q', 'MaxApertureValue:Q'])
#           )
                     
                     
    
    st.altair_chart(fig, use_container_width=True)
    
    st.markdown("""Looking at the ISO together with Aperture value also highlights the fact that he has composed shots to let in more light at higher ISO and wide open apertures.""")



    
    
    
if __name__ == '__main__':

    run()