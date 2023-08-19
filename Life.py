import pandas as pd
import plotly.figure_factory as ff
from urllib.request import urlopen
import json
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from Partisian import get_state_part_score
from sklearn.neighbors import KNeighborsClassifier


    
def get_age_categories():
    df = pd.read_excel(r"./data/Life/State/NY1.xlsx", header=2,skipfooter=1)
    df.rename(columns={ df.columns[0]: "age" }, inplace = True)
    age_categories = list(df["age"].values)
    return age_categories
    


def get_state_age_life_exp():
    all_states = []
    left = []
    center = []
    right = [] 
    age_categories = get_age_categories()
    state_score = get_state_part_score(2019)
    population = pd.read_csv("./data/Population/us-state-populations.csv").rename({"pop_2014":"pop"}, axis=1)
    
    
    for index, state in state_score.iterrows():
        state_name = state["State"]
        df = pd.read_excel(r"./data/Life/State/"+state_name+"1.xlsx", header=2,skipfooter=1)
        df.rename(columns={ df.columns[0]: "age" }, inplace = True)
        df["age_1"] = df["age"].str.replace(" ","–").str.split("–").str[0].astype(int)
        df["state"] = state_name
        all_states.append(df)
    all_states = pd.concat(all_states)
    all_states = state_score[["State","Part_Score","category"]].merge(all_states, left_on="State", right_on="state")
    all_states = all_states.merge(population, left_on="State",right_on="code")
    
    return all_states
    
    
def get_age_life_fig(metric = "qx"):
    all_states = get_state_age_life_exp()
    frames = []
    coef = []
    for k in range(100):
        age_df = all_states[all_states["age_1"]==float(k)]
        reg = LinearRegression().fit(age_df["Part_Score"].values.reshape(-1, 1), age_df[metric])
        X = np.linspace(-1,1)
        y = reg.predict(X.reshape(-1,1))
        #coef.append({"y":reg.coef_, "age":k})
        coef.append(float(reg.coef_))

        data = [go.Scatter(x=age_df["Part_Score"], 
                           y=age_df[metric], 
                           mode="markers", hovertext=age_df["State"],
                           marker=dict(color = age_df["Part_Score"], size=age_df["pop"]/500000, colorscale='bluered')),
                go.Scatter(x=X, 
                           y=y, 
                           mode="lines")
                ]
        
        if metric == "qx":
            highest = max(0.0005, age_df[metric].max())
            lowest = max(0, age_df[metric].min()-.005)
        else: 
            lowest = max(0, age_df[metric].max()-20000)
            highest =max(age_df[metric].max() + 1000, 20000)
        
        layout=go.Layout(
                xaxis={'title':f"Age : {k}",},
                yaxis ={"autorange":False, "range":[lowest,highest+.0005] },
                )
        frames.append(go.Frame(data=data, layout=layout))


    updatemenus = [
        dict(
            type='buttons',
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None,
                            dict(frame=dict(duration=240, redraw=True),
                                  transition=dict(
                                      duration=0,
                                      easing="linear" # <<=====
                                  ),
                                  fromcurrent=True,
                                  mode="immediate")])],
            direction="left",
            pad=dict(r=10, t=85),
            showactive=True, x=1.15, y=1, xanchor="right", yanchor="top")]


    fig = go.Figure(
        data=frames[0].data,
        layout=frames[0].layout,
        frames=frames
    )

    fig.update_layout(
      updatemenus=updatemenus, 
      # sliders=sliders
    )

    return fig
