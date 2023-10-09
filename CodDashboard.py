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
import GraphGenerator
from LifeData import LifeData
#from GraphGenerator import cultural_ontology
import time




app = JupyterDash(prevent_initial_callbacks="initial_duplicate")

app.config.suppress_callback_exceptions = True

app.layout = html.Div(className="row-fluid", children=[
    #html.H1(children='USA', style={'textAlign':'center'}),
    html.Div(className="container-fluid", id="graph-container", children=[]),
    html.Button('Submit', id='make-graph', n_clicks=0),
    
    dcc.Store(id='current_state')
])


@app.callback(Output('graph-container', 'children'),
              Input('make-graph', 'id'),
              State('graph-container', 'children'))
def change_main_view(start, parent):
    parent = add_dash_element(parent,  GraphGenerator.indicator_us_map("Life Exp"), True)
    parent = add_dash_element(parent, GraphGenerator.graph_placeholders("DeathVsPartisanship"), False)
    parent = add_dash_element(parent, GraphGenerator.graph_placeholders("CauseOfDeathTimeSeries"), False)
    parent = add_dash_element(parent, GraphGenerator.graph_placeholders("CauseOfDeathBoxPlot"), False)
    return parent


@app.callback(Output('graph-container', 'children', allow_duplicate=True),
              Output('current_state', 'data'),
              Input('StateWide', 'clickData'),
              Input('current_state', 'data'),
              State('graph-container', 'children'),
              prevent_initial_call=True)
def state_click(state_click, current_state, parent):
    state = "NationWide"
    if state_click:
        state = state_click["points"][0]["location"]
        #Load Json Code to State Conversion
        f = open("CodeToState.json")
        converter = json.load(f)
        state = converter[state]
        start_time = time.time()
        print(f"State: {state} Start Time: {start_time}")
        if current_state != state:
            parent = add_dash_element(parent, GraphGenerator.state_cod_tree_plot(state=state), True)
            parent = add_dash_element(parent, GraphGenerator.graph_placeholders("DeathVsPartisanship"), False)  
            parent = add_dash_element(parent, GraphGenerator.graph_placeholders("CauseOfDeathBoxPlot"), False)
            parent = add_dash_element(parent, GraphGenerator.graph_placeholders("CauseOfDeathTimeSeries"), False)
                 
        print(f"Total time: {time.time() - start_time}")
        
    return parent, state


@app.callback(Output('graph-container', 'children', allow_duplicate=True),
              Input('CauseOfDeathArea', 'clickData'),
              State('graph-container', 'children'), 
              prevent_initial_call=True)
def area_click(cod_click, parent):
    if cod_click:
        cod_label = cod_click["points"][0]["label"]
        try:
            state = cod_click["points"][0]["root"]
        except:
            pass
        if cod_label in GraphGenerator.get_ages():
            parent = add_dash_element(parent, GraphGenerator.partisan_comparison_age_cod(cod_label), False)
        else:
            cod, age = cod_label.split(": ")
            age = age.split("-")[0]
            parent = add_dash_element(parent, GraphGenerator.partisan_comparison_age_cod(age, cod), False)
            parent = add_dash_element(parent, GraphGenerator.all_ages_cod_boxplot(cod, state), False)
            parent = add_dash_element(parent, GraphGenerator.cod_timeseries(cod=cod,state=state), False)
        
    return parent
        

def add_dash_element(parent, element, new_row=True):
    #Find element (Different for tables vs graphs)
    try:
        new_id = element.id
    except:
        new_id = element["props"]["id"]
        print(new_id)
    #Check if graph already exists
    existing_graphs =find_child_ids(parent)

    if new_id in existing_graphs.keys():
       parent[existing_graphs[new_id]["location"][0]]["props"]["children"][existing_graphs[new_id]["location"][1]] = element
    #Create a new chart if id is new
    else:
        if new_row:
            new_row = html.Div(className="row", children=[]).to_plotly_json()
            new_row["props"]["children"].append(element)
            parent.append(new_row)
        else:
            parent[-1]["props"]["children"].append(element)
    return parent

def find_child_ids(parent):
    existing_graphs = {}
    for i, row in enumerate(parent):
        for j, child in enumerate(row["props"]["children"]):
            try:
                #Graph
                existing_graphs[child["props"]["id"]]  = {"location":[i,j]}
            except:
                #Table
                existing_graphs[child.id]  = {"location":[i,j]}
    return existing_graphs


if __name__ == '__main__':
    app.run_server(debug=True)