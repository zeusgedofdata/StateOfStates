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
import Partisian
import addfips


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
    df.drop(columns=["Notes", "State Code", ], inplace=True)
   
    try:
        f = open("data/Life/Deaths/ICD10Translation.json")
        translation = json.load(f)
        df["ICD-10 113 Cause List"] = df["ICD-10 113 Cause List"].apply(lambda x: translate_ICD10(x, translation))
        df.rename(columns={"ICD-10 113 Cause List":"Cause Of Death"}, inplace=True)
    except:
        None
    return df
    

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
        colorscale='Blues',
        hovertext=df["Indicator"],
        colorbar_x=0,
        marker_line_color='white'
    ), row=1, col=1)
    fig.update_layout(
        title_text =f"{df.Year.max()} {df.Indicator.unique()[0]} - State Wide",
        geo_scope='usa',
        height=520,
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
            
        )
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    
    
    #return fig
    graph_element = dcc.Graph(figure=fig, id="StateWide")
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
    year_income =  pd.read_csv(r"data\Life\health_ineq_online_table_2.csv")
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




def get_age_death(age):
    state_deaths = pd.read_csv(r"data/Life/Deaths/2019StateDeaths.txt", delimiter="	", na_values = ['Not Applicable'])
    df = clean_state_data(state_deaths)
    
    
    age_df = df[df["Ten-Year Age Groups Code"] == age]
    part = Partisian.get_state_part_score()
    age_df = pd.merge(age_df, part, left_on="State_Abv", right_on="State")
    X = age_df['Part_Score'].values.reshape(-1,1)
    y = age_df['Rate']
    
    reg = LinearRegression().fit(X, y, sample_weight=age_df['Population'])
    x_range = np.linspace(X.min(), X.max())
    y = reg.predict(x_range.reshape(-1,1))
    
    fig = make_subplots(rows=1, cols=2, vertical_spacing= 0.01, horizontal_spacing= 0.01,  subplot_titles=("",""), column_widths=[0.8, 0.2], shared_yaxes=True)
    
    fig.add_trace(go.Scatter(x=age_df['Part_Score'], y=age_df['Rate'],mode='markers', marker=dict(color=age_df["Part_Score"], colorscale="Bluered", size=age_df["Population"]/(age_df["Population"].max()/50)), name='Child', hovertext=age_df["State_x"]), row=1, col=1)
    
    hist_x = []
    for index, row in age_df.iterrows():
        hist_x.extend([row["Rate"] for i in range(int(round(row["Population"]/1000)))])
        row["Rate"]
    #fig.add_trace(go.Violin(y=hist_x, side="positive", name="Death Rate Distribution"), row=1, col=2)
    fig.add_trace(go.Violin(y=age_df['Rate'], side="positive",  name="Death Rate Distribution"), row=1, col=2)
    
    
    fig.add_trace(go.Scatter(x=x_range, y=y, mode='lines',  line=dict(color="green", dash="dash"), name='Best Fit Line'), row=1, col=1)
    
    fig.update_layout(title="Partisanship Score vs. Death Rate for " + age + " Year Olds", xaxis1_title="Partisanship Score", yaxis1_title="Death Rate (per 100,000)")
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=False)
    fig.update_layout(height=300,width=1200, plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=25, b=0))
    
    
    graph_element = dcc.Graph(figure=fig, id="DeathVsPartisanship" + age)
    dash_element = html.Div(graph_element, className="span6", id=f"DeathVsPartisanship{age}_parent")
    
    return dash_element

def load_state_cod(State = "Michigan"):
    f = open("icd-10-flat-Structure.json")
    flat_heirarchy = json.load(f)
    df_heirarchy = pd.DataFrame(flat_heirarchy)
    #rotate Matrix
    df_heirarchy = df_heirarchy.transpose()
    df_heirarchy.reset_index(inplace=True)
    df_heirarchy.rename(columns={"index":"ICD-10 113 Cause List"}, inplace=True)
    df_heirarchy

    df = pd.read_csv(r"data/Life/Deaths/StateDeathsAge.txt", delimiter="	", na_values = ['Not Applicable'])
    df = df.dropna(subset=["State","ICD-10 113 Cause List Code", "Population"]) 
    state_df= df[df["State"] == "Michigan"]
    #state_df = state_df[state_df["Ten-Year Age Groups Code"] ==age]
    state_df["Rate"] = (state_df["Deaths"] / state_df["Population"])*100000
    state_df["Rate"] = np.round(state_df["Rate"], 2)
    state_df["Child_Adj_Rate"] = state_df["Rate"]


    df_final = pd.merge(state_df, df_heirarchy, on="ICD-10 113 Cause List")
    df_final = df_final.drop(columns=["ICD-10 113 Cause List Code", "ranges", "chapters", "base", "Notes"])
    return df_final


def add_children(name, child, order, flat_heirarchy):
    order.append(name)
    if len(child["children"]) > 0: 
        for grand_child in child["children"]:
            add_children(grand_child, flat_heirarchy[grand_child], order, flat_heirarchy)
def get_depth_order(flat_heirarchy):
    order = []
    for key, value in flat_heirarchy.items():
        if len(value["parents"]) == 0:
            order.append(key)
            for children in value["children"]:
                add_children(children, flat_heirarchy[children],order, flat_heirarchy)
    return order


def get_age_cod(age, all_ages=False, state="Michgian"):
    df = load_state_cod()
    f = open("icd-10-flat-Structure.json")
    flat_heirarchy = json.load(f)
    order = get_depth_order(flat_heirarchy)
    df = df[df["Ten-Year Age Groups Code"] ==age]
    for e in range(5):
        for i, row in df.iterrows():
            children_total = 0
            children_values =[]
            if len(row["children"]) > 0:
                for child in row["children"]:
                    try:
                        children_total += df[df["ICD-10 113 Cause List"] == child]["Child_Adj_Rate"].values[0]
                        children_values.append(df[df["ICD-10 113 Cause List"] == child]["Child_Adj_Rate"].values[0])
                    except:
                        None
            df.at[i, "Child_Adj_Rate"] = max(row["Child_Adj_Rate"], children_total)
    df["ICD-10 113 Cause List"] = df["ICD-10 113 Cause List"].astype("category")
    df["ICD-10 113 Cause List"] = df["ICD-10 113 Cause List"].cat.set_categories(order, ordered=True)
    df = df.sort_values("ICD-10 113 Cause List")
    
    labels = [age]
    parents = [""]
    if all_ages:
        parents = ["All Ages"]
    values = [0]
    for i, row in df.iterrows():
        labels.append(row["ICD-10 113 Cause List"] + " " + str(age)[:2])
        if len(row["parents"]) > 0:
            parents.append(row["parents"][0] + " " + str(age)[:2])
        else: 
            parents.append(age)
            values[0] += row["Child_Adj_Rate"]
        values.append(row["Child_Adj_Rate"])
    return labels, parents, values, df, values[0]


def get_all_age_cod():
    df = load_state_cod()
    all_labels = ["All Ages"]
    all_parents = [""]
    all_values = [0]
    for age in df["Ten-Year Age Groups Code"].unique():
        labels, parents, values, df, total = get_age_cod(age, True)
        all_labels.extend(labels)
        all_parents.extend(parents)
        all_values.extend(values)
        all_values[0] += total
    return all_labels, all_parents, all_values