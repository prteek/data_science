#%%
import asyncio
from datetime import datetime
from shapely.geometry import MultiPolygon, Polygon
import pandas as pd
import lets_plot as gg
from lets_plot import LetsPlot, tilesets
from streamlit_letsplot import st_letsplot
import streamlit as st
import geopandas as gpd
import pgeocode
import os
from traveltimepy import Coordinates, PublicTransport, TravelTimeSdk
from datetime import datetime, timedelta
LetsPlot.setup_html()


#%%

def run():
    st.header("Public Transport Range")
    st.text("""Identify an area that is commutable within target time from the destination using 
public transport within Great Britain. Can be useful to identify where to rent  
based on travel times to work. There may be small pockets of regions further in 
distance but similar in time that may offer lower prices.
""")
    async def main():
        sdk = TravelTimeSdk(os.environ["TTPY_APPID"], os.environ["TTPY_APIKEY"])
        col1, col2 = st.columns(2)

        with col1:
            target_travel_time_mins = st.number_input("Target time (mins)", min_value=10, max_value=240)
        with col2:
            destination_postcode = st.text_input("Destination postcode", value='sw1a 1aa')
            nomi = pgeocode.Nominatim('gb')
            res = nomi.query_postal_code(destination_postcode.lower())

        ref = datetime.today() - timedelta(days=1)
        response = await sdk.time_map_geojson_async(
            coordinates=[Coordinates(lat=res.latitude, lng=res.longitude)],
            arrival_time=datetime(ref.year,ref.month,ref.day,9,0,0),
            travel_time=target_travel_time_mins*60,
            transportation=PublicTransport(type="public_transport"),
        )
        # print(response)
        return response

    response = asyncio.run(main())



    #%%
    geojson_data = response.features[0].geometry

    multipolygon_geom = MultiPolygon([Polygon(shell=coords[0], holes=coords[1:]) for coords in geojson_data.coordinates])
    # convert to a GeoDataFrame
    gdf = gpd.GeoDataFrame(pd.DataFrame({'geometry': [multipolygon_geom]}), geometry='geometry')

    # plot the GeoDataFrame
    p = gg.ggplot() + gg.geom_livemap(tiles=tilesets.OSM) + gg.geom_polygon(data=gdf, color='white', alpha=.7, size=.7) + gg.ggsize(500, 500)
    st_letsplot(p)

    #%%

