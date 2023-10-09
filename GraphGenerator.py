import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input, State, dash_table
from jupyter_dash import JupyterDash
import censusgeocode as cg
import plotly.figure_factory as ff
from urllib.request import urlopen
import json
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from Partisian import get_state_part_score
from Life import get_age_life_fig
from sklearn.neighbors import KNeighborsClassifier
from plotly.subplots import make_subplots
from math import floor
from LifeData import LifeData
import math
import Partisian as Partisian
import CauseOfDeathData
import addfips
import CleanData



states =  {'Alaska': 'AK','Alabama': 'AL','Arkansas': 'AR','American Samoa': 'AS','Arizona': 'AZ','California': 'CA','Colorado': 'CO','Connecticut': 'CT','District of Columbia': 'DC','Delaware': 'DE','Florida': 'FL','Georgia': 'GA','Guam': 'GU','Hawaii': 'HI','Iowa': 'IA','Idaho': 'ID','Illinois': 'IL','Indiana': 'IN','Kansas': 'KS','Kentucky': 'KY','Louisiana': 'LA','Massachusetts': 'MA','Maryland': 'MD','Maine': 'ME','Michigan': 'MI','Minnesota': 'MN','Missouri': 'MO','Northern Mariana Islands': 'MP','Mississippi': 'MS','Montana': 'MT','National': 'NA','North Carolina': 'NC','North Dakota': 'ND','Nebraska': 'NE','New Hampshire': 'NH','New Jersey': 'NJ','New Mexico': 'NM','Nevada': 'NV','New York': 'NY','Ohio': 'OH','Oklahoma': 'OK','Oregon': 'OR','Pennsylvania': 'PA','Puerto Rico': 'PR','Rhode Island': 'RI','South Carolina': 'SC','South Dakota': 'SD','Tennessee': 'TN','Texas': 'TX','Utah': 'UT','Virginia': 'VA','Virgin Islands': 'VI','Vermont': 'VT','Washington': 'WA','Wisconsin': 'WI','West Virginia': 'WV','Wyoming': 'WY'}


def get_best_fit(X, y):
    reg = LinearRegression().fit(X, y)
    x_range = np.linspace(X.min(), X.max())
    y = reg.predict(x_range.reshape(-1,1))
    return x_range, y[:,0], reg.coef_

def translate_ICD10(x, conversion):
    try:
        return conversion[x]
    except:
        return x
    
    
def get_ages():
    return  [0, 1, 5, 15, 25, 35, 45, 55, 65, 75, 85]
    

def clean_state_data(df):
    af = addfips.AddFIPS()
    try:
        df = df.dropna(subset=["State","ICD-10 113 Cause List Code", "Population"])
    except: 
        df = df.dropna(subset=["State", "Population"])
    #df = df[df["Crude Rate"] != "Unreliable"]
    df.Population = df.Population.astype(float)
    df["Rate"] = (df["Deaths"]/df["Population"])*100000
    df["Rate"] = df["Rate"].round(2)
    df["fips"] = df.apply(lambda x: af.get_state_fips(x["State"]), axis=1)
    df["Year"] = 2019
    df["State_Abv"] = df.apply(lambda x: states[x["State"]], axis=1)
    df = df.sort_values(by=["Rate"], ascending=False)
    try:
        df.drop(columns=["Notes", "State Code", ], inplace=True)
    except:
        None
   
    try:
        f = open("data/Life/Deaths/ICD10Translation.json")
        translation = json.load(f)
        df["ICD-10 Common"] = df["ICD-10 113 Cause List"].apply(lambda x: translate_ICD10(x, translation))
        df.rename(columns={"ICD-10 Common":"Cause Of Death"}, inplace=True)
    except:
        None
    return df



def graph_placeholders(id, span="span6"):
    return html.Div(id = id+"_parent", className=span)
    

def create_life_bar(indicator, state=None):
    life_data = LifeData()
    
    df = life_data.get_single_indicator(indicator)
    df = df.sort_values(by=["Rate"], ascending=False).reset_index(drop=True)
    colors  =['Blue',]*50
    if state:
        idx = df[df["State"]==state].index[0]
        colors[idx] = "Red"
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df["State"], x=df["Rate"], hovertext=df["Indicator"], marker_color=colors,  orientation='h'))
    min_x =round(math.floor(df["Rate"].min()/10))*10
    max_x=math.ceil(df["Rate"].max())
    fig.update_layout(
        xaxis_range=[min_x,max_x], 
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=600,
        )
    dash_element = html.Div(dcc.Graph(figure=fig, id="LifeBarChart"), className="span4", id="LifeBarChart_parent")
    return  dash_element
  
def life_us_map(indicator, state=None):
    life_data = LifeData()
    df = life_data.get_single_indicator(indicator)
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "choropleth"}]])
    
    fig.add_trace(go.Choropleth(
        locations = df["State"],
        z = df["Rate"],
        locationmode = 'USA-states',
        colorscale='RdBu',
        hovertext=df["Indicator"],
        colorbar_x=0,
        marker_line_color='white'
    ), row=1, col=1)
    fig.update_layout(
        title_text =f"{df.Year.max()} {df.Indicator.unique()[0]} - State Wide",
        geo_scope='usa',
        height=600,
        width=900,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    if state:
        fig.add_trace(go.Choropleth(
            locations = [state],
            z = [1],
            locationmode = 'USA-states',
            colorscale='Plasma',
            colorbar_x=0,
            marker_line_color='red',
            #remove color bar
            colorbar=dict(
                tickvals=[],
                ticks='',
                title=None
            )
        ), row=1, col=1)
        fig.update_layout(
            title_text =f"{df.Year.max()} {df.Indicator.unique()[0]} - {state}",
            xaxis={'fixedrange':True},
            yaxis={'fixedrange':True}
            
        )
    
    
    #return fig
    graph_element = dcc.Graph(figure=fig, id="StateWide", config={"scrollZoom": False, "displayModeBar": False})
    dash_element = html.Div(graph_element, className="span6", id="StateWide_parent")
    return dash_element




def state_card(state):
    life_data = LifeData()
    df = life_data.get_state_card_data(state)
    table  = dash_table.DataTable(
             id="StateCard", page_size=10, 
             data=df.to_dict('records'),
             style_cell={'textAlign': 'left'},
             style_as_list_view=True).to_plotly_json()
    title = html.H2(f"Health: {state}", style={"padding-top":"60px"})
    table_div = html.Div(id="StateCard_parent", children=[], className ="span2")
    table_div.children.append(title)
    table_div.children.append(table)
    return table_div




def cultural_ontology():
    df = pd.read_csv(r"./data/Groups/final_groups.csv")
    df.fips = df.fips.apply(lambda x: str(x).zfill(2))
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)
        
    fig = px.choropleth(df, geojson=counties, locations='fips', color='Cultural ID',
      color_continuous_scale="Viridis",
      range_color=(0, 12),
      scope="usa",
      #width=1200, height=400,
    )
    return fig


def create_scatter(df, indicator, fig, row, col, state):
    x_fit, y_fit, corr = get_best_fit(df[indicator].values.reshape(-1,1), (df["Infant Mortality"].values.reshape(-1,1)))
    
    fig.add_trace(go.Scatter(y = df["Infant Mortality"],  x = df[indicator], 
                                hovertext=df["State"],
                                mode="markers",    
                                marker=dict(
                                        size=6,
                                        color=df["Part_Score"], #set color equal to a variable
                                        colorscale='Bluered', # one of plotly colorscales
                                        #showscale=True
                                ),), row=row, col=col)
    
    fig.add_trace(go.Scatter(x = x_fit,  y = y_fit), row=row, col=col)
        
    fig.add_trace(go.Scatter(y=df[df["State"] == state]["Infant Mortality"], x = df[df["State"] == state][indicator],  mode="markers",  
                                 marker=dict(
                                        size=20,
                                        color="Green", 
                                        opacity=0.2,
                                ),), row=row, col=col)
    
    fig.add_hline(y=df[df["State"] == state]["Infant Mortality"].values[0], line_color="green", line_width=2, line_dash="dash", row=row, col=col, opacity=0.7)
    
    fig.update_yaxes(range=[3.5,8.5], row=row, col=col)


def birth_plots(state):
        #Get Data
        life_data = LifeData()
        pivot_state =life_data.get_long_format()
        part = Partisian.get_state_part_score()
        df = pd.merge(part, pivot_state)

        #Get selected state const
        state_infant_mortality = df[df["State"] == state]["Infant Mortality"].values[0]
        
        fig = make_subplots(rows=1, cols=4, vertical_spacing= 0.01, subplot_titles=("Infant Health Per 100k","","", ""))
        
        #hist = ff.create_distplot([df["Infant Mortality"]], ["Infant Mortality"])
        #
        #fig.add_trace(hist.data[0], row=1, col=1)
        #fig.add_trace(hist.data[1], row=1, col=1)
        fig.add_trace(go.Violin(y=pivot_state["Infant Mortality"],
                                name="Infant Mortality", 
                                side="negative", 
                                meanline_visible=True,
                                ),row=1,col=1)
        fig.add_hline(y=state_infant_mortality, line_color="green", line_width=2, line_dash="dash", row=1, col=1, opacity=0.7)
        fig.update_yaxes(range=[3.5,8.5], row=1, col=1)
        create_scatter(df, "Low Birth Weight", fig, 1, 2, state)
        create_scatter(df, "Premature", fig, 1, 3, state)
        create_scatter(df, "Part_Score", fig, 1, 4, state)


        fig.update_yaxes(showticklabels=False, row=1, col=3)
        fig.update_yaxes(showticklabels=False, row=1, col=4)
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=False)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=False)
        
        fig['layout']['yaxis1']['title']=''
        fig['layout']['xaxis1']['title']='Infant Mortality State Dist'
        fig['layout']['xaxis2']['title']='Low Birth Weight'
        fig['layout']['xaxis3']['title']='Premature Births'
        fig['layout']['xaxis4']['title']='Conservativeness'
        fig['layout']['yaxis1']['title']='Infant Mortality'
        
        fig.update_layout(height=300,width=1200, plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=0, b=0))
        
        
        #hide legend
        fig.update_layout(showlegend=False)
        #Add yaxis titles
        
        graph_element = dcc.Graph(figure=fig, id="BirthScatter")
        dash_element = html.Div(graph_element, className="span6", id="BirthScatter_parent")
        return dash_element
    
    
def create_life_scatter(df):
    male_year =df[df["gnd"] == "M"] 
    female_year =df[df["gnd"] == "F"] 
    data = [go.Scatter(x=male_year["hh_inc"], y=male_year["le_agg"], line_color="blue", mode="markers", name="Male"),
            go.Scatter(x=female_year["hh_inc"], y=female_year["le_agg"], line_color="red", mode="markers", name="Female"),
            go.Scatter(x=[0, 1000000], y = [male_year["le_agg"].mean(), male_year["le_agg"].mean()], mode="lines",name="Male Avg", line_color="#7694c4", line_dash="dash"),
            go.Scatter(x=[0, 1000000], y = [female_year["le_agg"].mean(), female_year["le_agg"].mean()], mode="lines",name="Female Avg", line_color="#c47680", line_dash="dash")]
    return data



def compare_male_female():
    year_income =  pd.read_csv(r"..\data\Life\health_ineq_online_table_2.csv")
    data = create_life_scatter(year_income[year_income["year"]==2001])
    
    layout = layout=go.Layout(
            xaxis_type="log",
            title="Year: 2001",
            xaxis=dict(range=[2, 6], autorange=False),
            yaxis=dict(range=[65, 92], autorange=False),
            #margin=dict(l=0, r=0, b=0, t=50),


            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                              method="animate",
                              args =  [None, {"frame": {"duration": 1000, "redraw": False},
                                    "fromcurrent": True, "transition": {"duration": 750,
                                                                        "easing": "linear"}}],)])])

    frames = []
    for year in year_income["year"].unique():
        frames.append(go.Frame(data = create_life_scatter(year_income[year_income["year"]==year])))
    fig =go.Figure(data=data, layout=layout, frames=frames)
    
    
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=50, b=0))
    fig['layout']['xaxis']['title']='House Hold Income - Age 40'
    fig['layout']['yaxis']['title']='Life Expectancy - Age 40'
    
    for i, frame in enumerate(fig.frames):
        frame['layout']['title'] = f"Year: {2001+i} "
        #redraw frame
    for button in fig.layout.updatemenus[0].buttons:
        button['args'][1]['frame']['redraw'] = True
    return fig

    graph_element = dcc.Graph(figure=fig, id="LifeIncomeScatter")
    dash_element = html.Div(graph_element, className="span6", id="LifeIncomeScatter_parent")
    
    return dash_element

def state_income_life():
    state_income_life_exp = pd.read_csv(r"data\Life\health_ineq_online_table_3.csv")
    state_income_life_exp = state_income_life_exp[state_income_life_exp["statename"]!="District of Columbia"]
    state_income_life_exp["all_q1"] = (state_income_life_exp["le_agg_q1_M"] + state_income_life_exp["le_agg_q1_F"])/2
    state_income_life_exp["all_q4"] = (state_income_life_exp["le_agg_q4_M"] + state_income_life_exp["le_agg_q4_F"])/2
    
    line_data = {"line_x": [], "line_y": []}
    for state in state_income_life_exp["statename"]:
        df = state_income_life_exp[state_income_life_exp["statename"]==state]
        line_data["line_x"].append(df["all_q1"].values[0])
        line_data["line_x"].append(df["all_q4"].values[0])
        line_data["line_x"].append(None)
        line_data["line_y"].extend([state, state, None])

    state_income_life_exp.sort_values(by="all_q1", inplace=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter( x = state_income_life_exp["all_q1"], y=state_income_life_exp["statename"], mode="markers", name = "Bottom 25%"))
    fig.add_trace(go.Scatter( x = state_income_life_exp["all_q4"], y=state_income_life_exp["statename"], mode="markers", name = "Top 25%"))
    fig.add_trace(go.Scatter( x = line_data["line_x"], y=line_data["line_y"], mode="lines", marker={"color":"grey"}, opacity=0.5,))
    fig.update_layout(width=800, plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=50, b=0))
    
    graph_element = dcc.Graph(figure=fig, id="StateIncomeLifeDumbell")
    dash_element = html.Div(graph_element, className="span6", id="StateIncomeLifeDumbell_parent")
    
    return dash_element



def get_state_cod_area(state = "Michigan"):
    final_labels, final_parents, final_values = CauseOfDeathData.get_all_age_cod(state)    
    
    fig = go.Figure(go.Treemap(
        branchvalues = "total",
        labels = final_labels,
        parents =  final_parents,
        values=final_values,
        textinfo = "label+value+percent entry",
        textposition = 'middle center',
        textfont = dict(family="Arial", size=16, color = '#FFFFFF'),
        pathbar_textfont_size=15,
        pathbar_visible=True,
    ))

    fig.update_layout(
        autosize=False,
        width=1850,
        height=350,
        
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=0
        ),)
    
    graph_element = dcc.Graph(figure=fig, id="CauseOfDeathArea" )
    dash_element = html.Div(graph_element, className="span12", id=f"CauseOfDeathArea_parent")
    
    return dash_element



def nation_single_cod(cod = "Major cardiovascular diseases (I00-I78)", state = "Michigan"):
    # Load Data
    df = pd.read_csv(r"..\data\Life\Deaths\2019CuaseOfDeath.txt", delimiter="	", na_values = ['Not Applicable'])
    df = clean_state_data(df)
    # Filter by COD
    df = df[df["ICD-10 113 Cause List"] == cod].sort_values("Rate", ascending=False)
    
    fig = go.Figure()
    fig.add_trace(go.Box(
        x=df["Rate"], 
        name=cod.split("(")[0],
        boxpoints='all', # can also be outliers, or suspectedoutliers, or False
        jitter=0.3, # add some jitter for a better separation between points
        pointpos=-1.8, # relative position of points wrt box
        hovertext=df["State_Abv"]

    ))
    fig.add_vline(x=df[df["State"]==state]["Rate"].values[0], line_width=3, line_dash="dash", line_color="green")
    
    fig.update_layout(margin = dict(t=50, l=10, r=0, b=0))
    fig.update_layout(
        autosize=False,
        width=800,
        height=200,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=0
        ),)
    

    graph_element = dcc.Graph(figure=fig, id="CauseOfDeathBoxPlot" )
    dash_element = html.Div(graph_element, className="span6", id=f"CauseOfDeathBoxPlot_parent")
    return dash_element



def get_age_death(age, cod="All"):

    df = CauseOfDeathData.load_state_cod(state="All", age=age, cod=cod)
    if cod == "All":
        df = df[df["parent_common_name"].isna()]
        df = df.groupby("state").agg({"deaths": "sum", "population":"mean", "partisan_control":"mean"}).reset_index()
        df["rate"] = (df["deaths"]/df["population"])*100000
        df["rate"] = df["rate"].round(0)
    
    df.dropna(subset=["partisan_control"], inplace=True)

    X = df['partisan_control'].values.reshape(-1,1)
    y = df['rate']
    
    reg = LinearRegression().fit(X, y, sample_weight=df['population'])
    x_range = np.linspace(X.min(), X.max())
    y = reg.predict(x_range.reshape(-1,1))
    
    fig = make_subplots(rows=1, cols=2, vertical_spacing= 0.01, horizontal_spacing= 0.01,  subplot_titles=("",""), column_widths=[0.8, 0.2], shared_yaxes=True)
    
    fig.add_trace(go.Scatter(
        x=df['partisan_control'], 
        y=df['rate'],
        mode='markers', 
        marker=dict(
            color=df["partisan_control"], 
            colorscale="Bluered", 
            size=df["population"]/(df["population"].max()/50)), 
        name='State', 
        hovertext=df["state"]
    ), row=1, col=1)
    
    hist_x = []
    for _, row in df.iterrows():
        hist_x.extend([row["rate"] for i in range(int(round(row["population"]/1000)))])
        row["rate"]
    fig.add_trace(go.Violin(y=df['rate'], side="positive",  name=""), row=1, col=2)
    
    
    fig.add_trace(go.Scatter(
        x=x_range, 
        y=y, 
        mode='lines',  
        line=dict(color="green", dash="dash"), 
        name='Best Fit Line'
    ), row=1, col=1)
    
    fig.update_layout(
        xaxis1_title=f"Partisanship Score vs. {cod[:25]} Mortality Rate for {age} Year Olds", 
        yaxis1_title="Mortality Rate (per 100,000)")
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=False)
    fig.update_layout(height=200,width=900, plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=25, b=0))
    
    graph_element = dcc.Graph(figure=fig, id="DeathVsPartisanship" + str(age))
    dash_element = html.Div(graph_element, className="span6", id=f"DeathVsPartisanship_parent")
    
    return dash_element


    
def get_age_box(cod, state="Michigan"):
    df = CauseOfDeathData.load_state_cod(state= "All", cod = cod)
    df["age"] = df["start_age"].astype(str) + "-" + df["end_age"].astype(str)
    df = df.sort_values(by=["start_age"])
    
    

    fig = px.box(df, x="age", y="rate", color="age", hover_data=["state", "common_name"])

    df = df[df["state"] == state]
    df = df.sort_values(by=["start_age"])
    fig.add_trace(go.Scatter(
        x = df["age"],
        y=df["rate"],
        mode = "markers+lines",
        marker=dict(color = "red"),
        name = state

    ))
    fig.update_layout(
        height=200,
        width=850,
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=25, b=0),)
    
    fig['layout']['xaxis']['title']=f"{cod} Mortality Rate By Age Group"
    graph_element = dcc.Graph(figure=fig, id="CauseOfDeathBoxPlot")
    dash_element = html.Div(graph_element, className="span6", id=f"CauseOfDeathBoxPlot_parent")

    return dash_element



def cod_timeseries(cod = "Suicide", state = "Michigan"):
    df = CauseOfDeathData.load_state_cod(state="All", year="All", cod = cod)
    df = df.sort_values(by=["year", "start_age"])
    state_df = df[df["state"] == state]
    
    national_avg = df.groupby(["year"]).agg({"deaths": "sum", "population": "sum"}).reset_index()
    national_avg["rate"] = national_avg["deaths"] / national_avg["population"] * 100000
    
    state_avg = state_df.groupby(["year"]).agg({"deaths": "sum", "population": "sum"}).reset_index()
    state_avg["rate"] = state_avg["deaths"] / state_avg["population"] * 100000
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = national_avg["year"],
        y = national_avg["rate"],
        mode = "lines",
        name = "National Avg",
        line=dict(
            width=4,
        )))

    fig.add_trace(go.Scatter(
        x = state_avg["year"],
        y = state_avg["rate"],
        mode = "lines",
        name = f"{state} Avg",
        line=dict(
            width=4,
        )))
    fig.update_layout(
        autosize=False,
        width=850,
        height=200,
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=0
        ),)
    fig['layout']['xaxis']['title']=f"{cod} Mortality Rate Over Time"
    graph_element = dcc.Graph(figure=fig, id="CauseOfDeathTimeSeries")
    dash_element = html.Div(graph_element, className="span6", id=f"CauseOfDeathTimeSeries_parent")

    return dash_element