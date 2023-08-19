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



app = JupyterDash(prevent_initial_callbacks="initial_duplicate")

app.config.suppress_callback_exceptions = True

app.layout = html.Div(className="row-fluid", children=[
    html.H1(children='USA', style={'textAlign':'center'}),
    html.Div(dcc.Dropdown(['State','Cultural Ontology'], 'State', id = "divide_dropdown"), className = "span8 offset2"),
    html.Div(className="container-fluid", id="graph-container", children=[]),
    #html.Div(className="container-fluid", id="graph-container-2", children=[]),
    html.Button('Submit', id='make-graph', n_clicks=0)
])


@app.callback(Output('graph-container', 'children'),
              Input('divide_dropdown', 'value'),
              State('graph-container', 'children'))
def change_main_view(divide_category, parent):
    if divide_category == "State":
        us_map = GraphGenerator.life_us_map("Life Exp")
    else:
        us_map = html.Div(dcc.Graph(figure=GraphGenerator.life_us_map("Life Exp"), id="New-ID"), className="span6", id="New-ID_parent")
    parent = add_dash_element(parent, us_map, True)
    return parent



@app.callback(Output('graph-container', 'children', allow_duplicate=True),
              Input('StateWide', 'clickData'),
              State('graph-container', 'children'),
              prevent_initial_call=True)
def state_click(state_click, parent):
    if state_click:
        state = state_click["points"][0]["location"]
        ind = state_click["points"][0]["hovertext"]
    

        #life_bar = GraphGenerator.create_life_bar(ind, state)
        #parent = add_dash_element(parent, life_bar, False)
        parent = add_dash_element(parent, GraphGenerator.state_card(state), False)
        parent =  add_dash_element(parent, GraphGenerator.life_us_map(ind, state), False)
        
        parent = add_dash_element(parent, GraphGenerator.compare_male_female(), True)
        parent = add_dash_element(parent,  GraphGenerator.state_income_life(), False)
        parent = add_dash_element(parent,  GraphGenerator.birth_plots(state), True)
        
    
    return parent

@app.callback(Output('graph-container', 'children', allow_duplicate=True),
              Input('StateCard', 'data'),
              Input('StateCard', 'active_cell'),
              State('graph-container', 'children'),
              prevent_initial_call=True)
def state_card_click(df, active_cell, parent):

    if active_cell:
        row = active_cell["row"]
        ind = df[row]["Indicator"]    
        state = df[row]["State"] 
        us_map = GraphGenerator.life_us_map(ind, state)
        birth_plots = GraphGenerator.birth_plots(state)

        parent = add_dash_element(parent, us_map, True)
        parent = add_dash_element(parent, birth_plots, True)
        
        #life_bar = GraphGenerator.create_life_bar(ind, state)
        #parent = add_dash_element(parent, life_bar, True)
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
            #new_row.children.append(element)
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