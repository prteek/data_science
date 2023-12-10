# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import holoviews as hv
from scipy import stats
from IPython.display import display
import duckdb
import requests
from bs4 import BeautifulSoup as BS
import imdb
import tqdm
from joblib import Parallel, delayed


hv.extension("bokeh")
sns.set_theme()

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

# %%
query_read_and_format_data = """
select *
from read_csv_auto('./docs/movies_dataset.csv') where runtime_mins is not null
"""

df_yearly_top_movies = sql(query_read_and_format_data)

assert (
    df_yearly_top_movies["runtime_mins"].isna().sum() == 0
), "Error in running time parsing"

# %%

kdims = [("release_year", "Release year")]
vdims = [("runtime_mins", "Runtime (mins)"), ("title", "Title")]
ds = hv.Dataset(df_yearly_top_movies, kdims=kdims, vdims=vdims)

annotate = [
    (2022, 192, "Avatar: The Way of Water"),
    (2019, 181, "Avengers: Endgame"),
    (2003, 201, "LOTR: The Return of the King"),
    (1993, 195, "Schindler's List"),
    (2014, 169, "Interstellar"),
]
annotations = hv.Overlay(
    [
        hv.Text(i[0] - 2, i[1] + 5, i[2], group="annotation_text").opts(
            text_font_size="12px"
        )
        for i in annotate
    ]
) * hv.Overlay(
    [
        hv.Scatter((i[0], i[1]), group="annotation_point").opts(
            line_color="red", fill_alpha=0
        )
        for i in annotate
    ]
)

agg = ds.aggregate("release_year", function=np.mean)
fig = (hv.Scatter(ds, group="data") * hv.Curve(agg, group="data") * annotations).opts(
    hv.opts.Curve(
        width=600,
        height=600,
        show_grid=True,
        color="k",
        title="Top grossing movies per year",
        xlim=(1988, 2025),
    ),
    hv.opts.Scatter(size=5, tools=["hover"]),
)

display(fig)


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

ds = hv.Dataset(
    df_test,
    kdims=[("release_year", "Release year")],
    vdims=[("runtime_mins", "Runtime (mins)")],
)

fig = (
    ds.to(hv.Distribution, "runtime_mins")
    .overlay("release_year")
    .opts(width=600, height=600, show_grid=True)
)

display(fig)

variable = "runtime_mins"
alpha = 0.05

g1 = sql(f"select {variable} from df_test where release_year != '{test_year}'")
g2 = sql(f"select {variable} from df_test where release_year = '{test_year}'")


def difference_of_mean(sample1, sample2):
    statistic = np.mean(sample1) - np.mean(sample2)
    return statistic


res = stats.bootstrap(
    (g1, g2), statistic=difference_of_mean, alternative="less", random_state=42
)

print(
    f"mean(difference of means) : {np.mean(res.bootstrap_distribution)}",
    "\n",
    f"proportion of samples < 0 : {np.mean(res.bootstrap_distribution < 0)}",
    "\n"
    f"Null hypothesis rejected: {(1 - np.mean(res.bootstrap_distribution < 0)) <= alpha}",
)


fig = (
    hv.Distribution(res.bootstrap_distribution[0])
    * hv.VLine(0).opts(
        color="black",
        xlabel="Bootstrap difference of means",
        width=600,
        height=600,
        show_grid=True,
    )
    * hv.Text(
        -15,
        0.02,
        f"proportion of \n samples < 0: {round(np.mean(res.bootstrap_distribution < 0),2)}",
    ).opts(text_font_size="12px")
    * hv.Text(
        -30,
        0.04,
        f"Null hypothesis rejected: {(1 - np.mean(res.bootstrap_distribution < 0)) <= alpha}",
    ).opts(text_font_size="12px")
    * hv.Text(-20, 0.001, f"mean: {round(np.mean(res.bootstrap_distribution),2)}").opts(
        text_font_size="12px"
    )
)

display(fig.opts(title="Testing difference between means of Runtimes"))


# %%
