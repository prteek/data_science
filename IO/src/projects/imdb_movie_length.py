# %%
import numpy as np
import pandas as pd
from scipy import stats
import duckdb
import requests
from bs4 import BeautifulSoup as BS
import imdb
import tqdm
from joblib import Parallel, delayed
import streamlit as st
from streamlit_letsplot import st_letsplot
import lets_plot as gg
from lets_plot import LetsPlot
LetsPlot.setup_html()

sql = lambda q: duckdb.sql(q).df()

# %%
# Scraping data

ia = imdb.Cinemagoer()


def get_year_matched_movie_from_title(title: str, year: int):
    """Since top titles are fetched from boxoffice mojo and movie details are fetched from imdb, the titles need to be matched to suitable year in imdb database (due to movies with same name)"""
    movies = ia.search_movie(title)
    for movie in movies:
        if ia.get_movie_main(movie.getID())["data"]["year"] == year:
            return movie
        else:
            continue
    return None


def get_info_from_movie(movie):
    """This function can be modified to include desired info from movie object collected as a dictionary"""
    run_time = ia.get_movie_main(movie.getID())["data"]["runtimes"][0]
    year = ia.get_movie_main(movie.getID())["data"]["year"]
    title_info = {
        "release_year": year,
        "runtime_mins": int(run_time),
    }
    return title_info


def get_info_for_title(title, year):
    """This function exists solely to package functionality together and enbale use of Parellelism"""
    movie = get_year_matched_movie_from_title(title, year)
    if movie is not None:
        title_info = get_info_from_movie(movie)
    else:
        title_info = dict()
        title_info["release_year"] = year
        # Comment statement below since column will be autofilled
        # title_info["runtime_mins"] = np.nan

    title_info["title"] = title
    return title_info


def download_data():
    years = range(1990, 2024)
    yearly_top_grossing_url = "https://www.boxofficemojo.com/year/world/{year}/"

    top_n = 10
    all_titles = []
    pbar = tqdm.tqdm(years, position=0)
    problematic_movie_titles = [
        "300",
        "Fast & Furious 6",
    ]
    # 300 was probably released in 2006 but appears on charts in 2007 and isn't matched correctly
    # Fast & Furious 6 has incorrect runtime duration (19 min)

    for year in pbar:
        pbar.set_description(str(year))
        page = requests.get(yearly_top_grossing_url.format(year=year))
        soup = BS(page.content, "html.parser")
        titles = soup.find_all("td", class_="a-text-left mojo-field-type-release_group")
        delayed_year_results = []
        for t in titles[:top_n]:
            title = t.select("a")[0].string
            if title in problematic_movie_titles:
                continue
            title_info = delayed(get_info_for_title)(title, year)
            delayed_year_results.append(title_info)

        year_results = Parallel(n_jobs=top_n, prefer="threads")(delayed_year_results)
        all_titles.extend(year_results)

    df_movies = pd.DataFrame(all_titles)
    df_movies.to_csv("./docs/movies_dataset.csv", index=False)
    return df_movies


# %%

def run():
    query_read_and_format_data = """
    select *
    from read_csv_auto('https://raw.githubusercontent.com/prteek/data_science/main/IO/src/projects/docs/movies_dataset.csv') where runtime_mins is not null
    """

    df_yearly_top_movies = sql(query_read_and_format_data)

    assert (
        df_yearly_top_movies["runtime_mins"].isna().sum() == 0
    ), "Error in running time parsing"

    #%%

    # df_yearly_top_movies['release_year'] = pd.to_datetime(df_yearly_top_movies['release_year'], format='%Y')

    annotate = [
        {"release_year": 2022, "runtime_mins": 192, "title": "Avatar: The Way of Water"},
        {"release_year": 2019, "runtime_mins": 181, "title": "Avengers: Endgame"},
        {"release_year": 2003, "runtime_mins": 201, "title": "LOTR: The Return of the King"},
        {"release_year": 1993, "runtime_mins": 195, "title": "Schindler's List"},
        {"release_year": 2014, "runtime_mins": 169, "title": "Interstellar"},
    ]

    df_annotations = pd.DataFrame(annotate)
    mean_df = df_yearly_top_movies.groupby('release_year')['runtime_mins'].mean().reset_index()

    p1 = gg.ggplot(df_yearly_top_movies, gg.aes(x='release_year', y='runtime_mins')) + \
        gg.geom_point(size=5, tooltips=gg.layer_tooltips().line('@{title}')) + \
        gg.geom_text(gg.aes(label='title'), data=df_annotations, color="red", nudge_y=5) + \
        gg.geom_line(gg.aes(y='runtime_mins'), data=mean_df, color="black") + \
        gg.ggtitle('Top grossing movies per year')


    # %%

    base_window = 1990, 2001
    test_year = 2022

    # We shall try to create a tidy dataset for further exploration
    query_test = f"""
    select runtime_mins
    , case when release_year={test_year} then '{test_year}'
    else '{base_window[0]}-{base_window[1]-1}'
    end as release_year
    from df_yearly_top_movies
    where (release_year >= {base_window[0]} and release_year < {base_window[1]}) or (release_year={test_year})
    """

    df_test = sql(query_test)

    p2 = gg.ggplot(df_test) + \
        gg.geom_density(
            gg.aes(x='runtime_mins', fill='release_year'),
            alpha=0.5,
            size=0.5
        )


    #%%

    variable = "runtime_mins"
    alpha = 0.05

    g1 = sql(f"select {variable} from df_test where release_year != '{test_year}'")
    g2 = sql(f"select {variable} from df_test where release_year = '{test_year}'")

    def difference_of_mean(sample1, sample2):
        statistic = np.mean(sample2) - np.mean(sample1)
        return statistic

    res = stats.bootstrap(
        (g1, g2), statistic=difference_of_mean, alternative="greater", random_state=42
    )

    print(
        f"mean(difference of means) : {np.mean(res.bootstrap_distribution)}",
        "\n",
        f"proportion of samples > 0 : {np.mean(res.bootstrap_distribution > 0)}",
        "\n"
        f"Null hypothesis rejected: {(1 - np.mean(res.bootstrap_distribution > 0)) <= alpha}",
    )

    data = {"x": res.bootstrap_distribution[0]}
    df = pd.DataFrame(data)

    vlines = [
        {"x_intercept": 0},
    ]

    labels = [
        {"x": 15, "y": 0.02,
         "label": f"proportion of \n samples > 0: {round(np.mean(res.bootstrap_distribution[0] > 0), 2)}"},
        {"x": 30, "y": 0.04,
         "label": f"Null hypothesis rejected: {(1 - np.mean(res.bootstrap_distribution[0] > 0)) <= alpha}"},
        {"x": 20, "y": 0.001, "label": f"mean: {round(np.mean(res.bootstrap_distribution[0]), 2)}"},
    ]

    df1 = pd.DataFrame(vlines)
    df2 = pd.DataFrame(labels)

    p3 = gg.ggplot(df, gg.aes(x='x')) + \
        gg.geom_density() + \
        gg.geom_vline(gg.aes(xintercept='x_intercept'), data=df1, color='black') + \
        gg.geom_text(mapping=gg.aes(x='x', y='y', label='label'), data=df2) + \
        gg.labs(x='Bootstrap difference of means')


    st.header("Are movies getting longer ?")
    st.markdown("""
I recently came across an article posing a question, ``` are blockbusters getting (reely) longer ? ```
Partly fuelled by the fact that world is going crazy over Christopher Nolan’s Oppenheimer and that this would be his longest movie (just over 3 hours).   
As an exercise in inference this question can be answered with some data.  
We can fetch top grossing movies in each year from *boxoffice mojo* and movie details from *imdb*.  
[code](https://github.com/prteek/data_science/blob/main/IO/src/projects/imdb_movie_length.py), 
[data](https://raw.githubusercontent.com/prteek/data_science/main/IO/src/projects/docs/movies_dataset.csv)
""")

    st_letsplot(p1)
    st.markdown("""
We can compare mean of movie runtimes from top grossing movies released between 1990-2000 to mean of runtimes from top grossing movies released in 2022.  
""")
    st_letsplot(p2)

    st.markdown("""If we bootstrap the mean of movie times and take the difference between means of release year 2022 from release years 1990-2000, 
there is evidence that blockbuster movies are getting longer in recent years, and the difference can be up to 20 min.  
[code](https://github.com/prteek/data_science/blob/main/IO/src/projects/imdb_movie_length.py), 
""")
    st_letsplot(p3)

    st.markdown("""
So on your next visit to the theatre make sure to get some extra popcorn.  
""")

    # %%
