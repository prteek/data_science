#%%
import numpy as np
import lets_plot as gg
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson
import streamlit as st
import pymc as pm
import pytensor as pt
import arviz as az
from lets_plot import LetsPlot
from streamlit_letsplot import st_letsplot
LetsPlot.setup_html()
sql = lambda q: duckdb.sql(q).df()

#%%

def run():
    results_raw = sql("""select * 
                        from read_csv_auto('https://raw.githubusercontent.com/prteek/data_science/main/IO/src/projects/docs/football_results.csv', sample_size=-1) 
                        where home_score <> 'NA' -- remove future fixtures that have no scores yet
                        """)

    # Teams in Europe
    teams = ('Albania',
        'Andorra',
        'Armenia',
        'Austria',
        'Azerbaijan',
        'Belarus',
        'Belgium',
        'Bosnia and Herzegovina',
        'Bulgaria',
        'Croatia',
        'Cyprus',
        'Czech Republic',
        'Denmark',
        'England',
        'Estonia',
        'Faroe Islands',
        'Finland',
        'France',
        'Georgia',
        'Germany',
        'Gibraltar',
        'Greece',
        'Hungary',
        'Iceland',
        'Republic of Ireland',
        'Israel',
        'Italy',
        'Kazakhstan',
        'Kosovo',
        'Latvia',
        'Liechtenstein',
        'Lithuania',
        'Luxembourg',
        'Malta',
        'Moldova',
        'Montenegro',
        'Netherlands',
        'North Macedonia',
        'Northern Ireland',
        'Norway',
        'Poland',
        'Portugal',
        'Romania',
        'Russia',
        'Scotland',
        'Serbia',
        'Slovakia',
        'Slovenia',
        'Spain',
        'Sweden',
        'Switzerland',
        'Turkey',
        'Ukraine',
        'Wales')


    start_date = '2021-07-12'
    end_date = '2024-06-10'

    results = sql(f"""
                  select cast(date as date) as date
                  , cast(home_score as numeric) as home_score 
                  , cast(away_score as numeric) as away_score
                  , home_team
                  , away_team
                  from results_raw
                  where 1=1
                  and (home_team in {teams} and away_team in {teams})
                  and date between cast('{start_date}' as date) and cast('{end_date}' as date) -- between Euro 2020 end and Euro 2024 start
    --               and date between cast('2018-07-16' as date) and cast('2021-05-28' as date)
                  order by date
                  """
                  )

    # assert len(teams) <= results['home_team'].nunique(), "Incorrect number of teams in data"
    # assert len(teams) <= results['away_team'].nunique(), "Incorrect number of teams in data"

    sql("select * from results limit 10")

    #%%

    # EDA

    print(sql("""select max(away_score)
                         , max(home_score)
                         , count(*) as total_games
                         , max(home_score-away_score)
                         , min(home_score-away_score)
                         from results
            """).T)

    print(sql("select home_team, count(*) from results group by 1 order by 2 desc limit 5"))
    print(sql("select home_team, sum(home_score) from results group by 1 order by 2 desc limit 5"))

    #%%

    dates = results['date'].values.astype('datetime64[D]')
    home_teams = results['home_team'].values
    away_teams = results['away_team'].values
    home_sc = results['home_score'].values
    away_sc = results['away_score'].values
    relevant_teams = np.array(teams)
    # Matrix showing games played between each team:
    games_played = np.zeros((len(relevant_teams), len(relevant_teams)))

    # Matrix where (i,j) entry is goals scored by team i against team j across all matches:
    goals_scored_matrix = np.zeros((len(relevant_teams), len(relevant_teams)))
    goals_scored = np.zeros(len(relevant_teams))
    goals_conc = np.zeros(len(relevant_teams))

    indices = np.arange(len(relevant_teams))
    n = 0
    end_n = len(results)
    while n < end_n:
        if sum((relevant_teams == home_teams[n]) + (relevant_teams == away_teams[n])) < 2:
            n = n + 1
        else:
            # Finding indices of teams
            index_1 = indices[relevant_teams == home_teams[n]][0]
            index_2 = indices[relevant_teams == away_teams[n]][0]
            # Updating matrices
            games_played[index_1][index_2] = games_played[index_1][index_2] + 1
            games_played[index_2][index_1] = games_played[index_2][index_1] + 1
            goals_scored_matrix[index_1][index_2] = goals_scored_matrix[index_1][index_2] + home_sc[n]
            goals_scored_matrix[index_2][index_1] = goals_scored_matrix[index_2][index_1] + away_sc[n]
            goals_scored[index_1] = goals_scored[index_1] + home_sc[n]
            goals_scored[index_2] = goals_scored[index_2] + away_sc[n]
            goals_conc[index_1] = goals_conc[index_1] + away_sc[n]
            goals_conc[index_2] = goals_conc[index_2] + home_sc[n]
            n = n + 1

    check_count = 1
    P = games_played
    GS = goals_scored_matrix

    from scipy.optimize import fsolve
    f = goals_scored
    c = goals_conc
    P_zeros = np.zeros_like(games_played)
    Objective = lambda x: abs(np.matmul(np.block([[P, P_zeros], [P_zeros, P]]), x)*np.concatenate(
        [x[54:108], x[0:54]])  - np.concatenate([f, c]))
    Values = np.concatenate([f / np.sum(P, axis=0), c / np.sum(P, axis=0)])
    i = 0
    while i < 3:
        Values = fsolve(Objective, Values)
        print(np.sum(Objective(Values) ** 2))
        Attacks = Values[54:108]
        Defences = Values[0:54]
        i += 1

    #%%
    with pm.Model() as model:
        attacks = pm.Gamma('attacks', alpha=1, beta=10, shape=len(teams))
        defences = pm.Gamma('defences', alpha=1, beta=10, shape=len(teams))
        x = pt.tensor.concatenate([defences, attacks])
        mu = abs(pt.tensor.matmul(np.block([[P, P_zeros], [P_zeros, P]]), x) * pt.tensor.concatenate([attacks, defences]))
        scores = pm.Poisson('scores', mu=mu, observed=np.concatenate([f,c]))
        samples = pm.sample()

    Attacks = samples['posterior']['attacks'].values.mean(axis=0).mean(axis=0)
    Defences = samples['posterior']['defences'].values.mean(axis=0).mean(axis=0)

    #%%
    res = pd.DataFrame(np.c_[Attacks, Defences], columns=['attack', 'defence'])
    res.index = relevant_teams
    res['defence_scaled'] = res['defence']/res.loc['Gibraltar']['defence']  # Weakest defence
    res['attack_scaled'] = res['attack']*res.loc['Gibraltar']['defence']
    print(res)

    res.plot(x='attack_scaled', y='defence_scaled', grid=True, kind='scatter')
    plt.show()
    #%%
    def plot_dist(team, opponent):
        mu_1 = res.loc[team]['attack']*res.loc[opponent]['defence']
        mu_2 = res.loc[opponent]['attack']*res.loc[team]['defence']

        x = np.arange(0,6)
        pmf_1 = poisson.pmf(x, mu_1)
        pmf_2 = poisson.pmf(x, mu_2)

        # Plot
        plt.plot(x, pmf_1, 'bo', ms=8, label=team)
        plt.vlines(x, 0, pmf_1, colors='b', lw=5, alpha=0.5)
        plt.plot(x, pmf_2, 'ro', ms=8, label=opponent)
        plt.vlines(x, 0, pmf_2, colors='r', lw=5, alpha=0.5)
        plt.legend()
        plt.xlabel('Number of Goals')
        plt.ylabel('Probability')
        plt.title(f"{team}: {round(mu_1,2)}, {opponent}: {round(mu_2,2)}")
        plt.grid()
        plt.show()
        return None


    def calculate_goals_matrix(team_mu, opponent_mu, n_goals=5):
        x = np.arange(0, n_goals+1)
        pmf_1 = poisson.pmf(x, team_mu)
        pmf_2 = poisson.pmf(x, opponent_mu)
        goals_matrix = np.outer(pmf_1, pmf_2)/np.sum(np.outer(pmf_1, pmf_2))
        return goals_matrix

    def plot_bivariate(team, opponent):
        n_goals = 5
        team_mu = res.loc[team]['attack'] * res.loc[opponent]['defence']
        opponent_mu = res.loc[opponent]['attack'] * res.loc[team]['defence']
        goals_matrix = calculate_goals_matrix(team_mu, opponent_mu, n_goals)
        x = np.arange(0, n_goals+1)
        # Create DataFrame
        df = pd.DataFrame({team: np.repeat(x, len(x)),
                           opponent: np.tile(x, len(x)),
                           'probability': goals_matrix.flatten()})
        p = gg.ggplot(df, gg.aes(x=team, y=opponent, fill='probability')) + \
            gg.geom_tile() + \
            gg.scale_x_continuous(breaks=x) + \
            gg.scale_y_continuous(breaks=x) + \
            gg.ggtitle(f"{team}:{round(team_mu,2)}, {opponent}:{round(opponent_mu,2)}")
        p.show()
        return p

    #%%
    # Group predictions
    groups = [
        ('Germany', 'Switzerland', 'Hungary', 'Scotland')
        , ('Spain', 'Italy', 'Albania', 'Croatia')
        , ('England', 'Denmark', 'Slovenia', 'Serbia')
        , ('Netherlands', 'France', 'Poland', 'Austria')
        , ('Romania', 'Slovakia', 'Belgium', 'Ukraine')
        , ('Turkey', 'Portugal', 'Czech Republic', 'Georgia')
    ]
    group_names = ['A', 'B', 'C', 'D', 'E', 'F']

    points_map = [3, 0, 1]
    league_table = []
    for group, group_name in zip(groups, group_names):
        group_results = pd.DataFrame(np.zeros((len(group),len(group))), columns=group)
        group_results.index = group
        group_results_goals = pd.DataFrame(np.zeros((len(group),len(group))), columns=group)
        group_results_goals.index = group
        for team in group:
            for opponent in group:
                if opponent == team: continue
                n_goals = 5
                team_mu = res.loc[team]['attack'] * res.loc[opponent]['defence']
                opponent_mu = res.loc[opponent]['attack'] * res.loc[team]['defence']
                goals_matrix = calculate_goals_matrix(team_mu, opponent_mu, n_goals)
                index_2d = np.unravel_index(np.argmax(goals_matrix), goals_matrix.shape)
                # scenario = np.argmax([np.sum(np.tril(goals_matrix, k=-1)), np.sum(np.triu(goals_matrix, k=1)), np.sum(np.diag(goals_matrix))])
                # points = points_map[scenario]
                points = (index_2d[0] > index_2d[1])*3 + (index_2d[0] == index_2d[1])
                group_results.loc[team, opponent] = points
                group_results_goals.loc[team, opponent] = index_2d[0]

        group_table = group_results.sum(axis=1).sort_values(ascending=False).reset_index().rename({'index': 'team', 0: 'points'}, axis=1)
        group_results_goals_table_for = group_results_goals.sum(axis=1).sort_values(ascending=False).reset_index().rename({'index': 'team', 0: 'gf'}, axis=1)
        group_results_goals_table_against = group_results_goals.sum(axis=0).sort_values(ascending=False).reset_index().rename({'index': 'team', 0: 'ga'}, axis=1)
        group_table = group_table.merge(group_results_goals_table_for.merge(group_results_goals_table_against, on='team'), on='team')
        multi_col = pd.MultiIndex.from_product([[group_name], group_table.columns])
        group_table.columns = multi_col
        league_table.append(group_table)

    league_table = pd.concat(league_table, axis=1)

    #%%
    st.header('Euro 2024 model')
    st.caption("Predictions as on 10 June 2024")
    st.subheader('Group standings')
    cols = st.columns(2)
    for i, group_name in enumerate(group_names):
        group_table = league_table[group_name]
        with cols[i%len(cols)]:
            st.dataframe(group_table)

    st.markdown("---")
    st.subheader('Match predictor')
    euro_teams = [i for group in groups for i in group]
    col1, col2 = st.columns(2)
    with col1:
        team = st.selectbox("Select team", euro_teams, index=8)
    with col2:
        opponent = st.selectbox("Select opponent", euro_teams, index=10)

    p = plot_bivariate(team, opponent)
    st_letsplot(p);

    st.markdown("---")
    st.markdown("""
The model is a Double Poisson model (first developed 1982) , where goals scored by each team are assumed to be Poisson distributed with a mean depending on attacking and defensive strengths.
This model is simplistic yet insightful for the curious. Major shortcomings in the model are:  
1. Assuming that there are effectively 2 matches going on at each end independently
2. No accounting for the sequence of goals i.e. if Germany did score 2 goals first against Scotland, definitely coming back from that is hard
3. No effect of chain of previous games and essentially each next game is considered independent of the previous one
4. High weightage in strength calculation to games against very weak teams. Effectively due to more goals score in just 1 game inflate the overall strength value

The model relies on the assumption that goals are scored according to a Poisson Process, an assumption seen in a wide variety of papers.  
It is also more accurate in longer run prediction (i.e. across several matches) than any given outcome itself due to strength and weakness 
being a representation of long run performance of the team.  
HILL(1974) showed that football experts were able, before the season started, to predict with some success the final league table positions.  
Therefore, certainly over a whole season, skill rather than chance dominates the game. This would probably be agreed by most people who watch the game of football; 
that whilst in a single match, chance plays a considerable role (missed scoring opportunities, dubious offside deci- sions and shots hitting the crossbar can obviously drastically affect the result), 
over several matches luck plays much less ofa part.   

There are good reasons for thinking that the number of goals scored by a team in a match is likely to be a Poisson variable: possession is an important aspect of football, 
and each time a team has the ball it has the opportunity to attack and score. 
The probability p that an attack will result in a goal is, of course, small, but the number of times a team has possession during a match is very large.  
If p is constant and attacks are inde- pendent, the number of goals will be Binomial and in these circumstances the Poisson approximation will apply very well.
""")

    #%%