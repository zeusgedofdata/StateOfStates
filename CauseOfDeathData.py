import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input
import censusgeocode as cg
import plotly.figure_factory as ff
from urllib.request import urlopen
import json
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from Partisian import get_state_part_score
from Life import get_age_life_fig, get_state_age_life_exp
from sklearn.neighbors import KNeighborsClassifier
from plotly.subplots import make_subplots
import math
from LifeData import LifeData
import Partisian as Partisian
from sklearn.linear_model import LinearRegression
import addfips


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
        f = open(r"D:\StateOfStates\data\Life\Deaths\ICD10Translation.json")
        translation = json.load(f)
        df["ICD-10 Common"] = df["ICD-10 113 Cause List"].apply(lambda x: translate_ICD10(x, translation))
        df.rename(columns={"ICD-10 Common":"Cause Of Death"}, inplace=True)
    except:
        None
    return df
    


def top_n(df, n):
    #Filter out low occuring chapters
    df = df.groupby("ICD-10 113 Cause List Code").filter(lambda x: len(x)>1)
    df = df.sort_values(by = ["Rate"], ascending = False)
    df = df.head(n).copy()
    top_codes = list(df["ICD-10 113 Cause List Code"].unique())
    df = df[df["ICD-10 113 Cause List Code"].isin(top_codes)]
    return df

def get_heirarchy():
    '''Returns a dictionary of the ICD-10 heirarchy'''
    flat_heirarchy = json.load(open(r"D:\StateOfStates\icd-10-flat-Structure.json"))
    return flat_heirarchy

def get_heirarchy_df():
    '''Returns a dataframe of the ICD-10 heirarchy'''
    flat_heirarchy = json.load(open(r"D:\StateOfStates\icd-10-flat-Structure.json"))
    df_heirarchy = pd.DataFrame(flat_heirarchy)
    return df_heirarchy


def load_state_cod(state="Michgian"):
    '''Returns a dataframe of cause of death data for a single state broken up by age group'''
    df_heirarchy = get_heirarchy_df()
    #rotate Matrix
    df_heirarchy = df_heirarchy.transpose()
    df_heirarchy.reset_index(inplace=True)
    df_heirarchy.rename(columns={"index":"ICD-10 113 Cause List"}, inplace=True)
    df_heirarchy

    df = pd.read_csv(r"D:/StateOfStates/data/Life/Deaths/StateDeathsAge.txt", delimiter="	", na_values = ['Not Applicable'])
    df = df.dropna(subset=["State","ICD-10 113 Cause List Code", "Population"]) 
    
    #state_df = state_df[state_df["Ten-Year Age Groups Code"] ==age]
    df["Rate"] = (df["Deaths"] / df["Population"])*100000
    df["Rate"] = np.round(df["Rate"], 2)
    df["Child_Adj_Rate"] = df["Rate"]
    if(state != "All"):
        df= df[df["State"] == state]


    df_final = pd.merge(df, df_heirarchy, on="ICD-10 113 Cause List")
    df_final = df_final.drop(columns=["ICD-10 113 Cause List Code", "ranges", "chapters", "base", "Notes"])
    return df_final


def add_children(name, child, order, flat_heirarchy):
    order.append(name)
    if len(child["children"]) > 0: 
        for grand_child in child["children"]:
            add_children(grand_child, flat_heirarchy[grand_child], order, flat_heirarchy)
            
def get_depth_order():
    flat_heirarchy = get_heirarchy()
    order = []
    for key, value in flat_heirarchy.items():
        if len(value["parents"]) == 0:
            order.append(key)
            for children in value["children"]:
                add_children(children, flat_heirarchy[children],order, flat_heirarchy)
    return order



def get_age_cod(df, age, all_ages=False, state="Michigan"):
    '''Returns a list of labels, parents, and values for a given state and age group'''
    order = get_depth_order()
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
        parents = [state]
    values = [0]
    colors = ["white"]
    for i, row in df.iterrows():
        labels.append(row["ICD-10 113 Cause List"].split("(")[0] + ": " + str(age))
        if len(row["parents"]) > 0:
            parents.append(row["parents"][0].split("(")[0] + ": " + str(age))
        else: 
            parents.append(age)
            values[0] += row["Child_Adj_Rate"]
        values.append(row["Child_Adj_Rate"])
    return labels, parents, values, df, values[0]
    
def get_all_age_cod(state = "Michigan"):
    '''Returns a list of labels, parents, and values for a given state and all age groups'''
    df = load_state_cod(state = state)
    all_labels = [state]
    all_parents = [""]
    all_values = [0]
    for age in df["Ten-Year Age Groups Code"].unique():
        labels, parents, values, df_age, total = get_age_cod(df, age, True, state)
        all_labels.extend(labels)
        all_parents.extend(parents)
        all_values.extend(values)
        all_values[0] += total
    return all_labels, all_parents, all_values