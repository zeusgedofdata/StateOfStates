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
import psycopg2
import sqlalchemy
from sqlalchemy import create_engine, text




def get_db_properties():
    props = {}
    separator = "="
            
    with open(r"C:\Users\zeusg\OneDrive\Documents\Github\habit_final\database.properties") as file:
        for line in file: 
            if separator in line:
                name, value = line.split(separator, 1)
                props[name.strip()] = value.strip()
    return props

def get_db_connection():
    props = get_db_properties()
    url = f'postgresql+psycopg2://{props["username"]}:{props["password"]}@{props["host"]}:5432/StateOfStates'
    connection = sqlalchemy.create_engine(url)
    return connection

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


def load_state_cod_timeseries(state="All", age = "All", cod="All"):
    sql = f"select l.name as state, l.acronym, l.fips, \
            cod.rate, cod.deaths, cod.population, cod.start_age, cod.end_age, cod.year, \
            ic.name, ic.common_name, \
            ic2.name as parent_name, ic2.common_name as parent_common_name, \
            p.avg as partisan_control \
            from causes_of_death cod \
            inner join icd_10_codes ic on cod.icd_10_codes_id_fkey = ic.id \
            left join icd_10_codes ic2 on ic.parent = ic2.id \
            inner join locations l on cod.locations_id_fkey = l.id \
            inner join (select l.id, AVG(p.control_score) from locations l  \
				inner join partisans p on l.id=p.locations_id_fkey \
				where p.year < 2020 \
				group by l.id ) p on cod.locations_id_fkey =p.id \
            where l.level = 'State'"
            
    if state != "All":
        sql += "and l.name = '" + state + "'"
    if age != "All":
        sql += " and cod.start_age = '" + str(age) + "'"
    if cod != "All":
        sql += " and ic.common_name = '" + cod + "'"
    con = get_db_connection()
    df = pd.read_sql(sql=text(sql), con = con.engine.connect())
    
    return df



def load_state_cod(state="All", age = "All", cod="All", year=2019):
    '''Returns a dataframe of cause of death data for a single state broken up by age group'''
    sql = f"select l.name as state, l.acronym, l.fips, \
            cod.rate, cod.deaths, cod.population, cod.start_age, cod.end_age, cod.year, \
            ic.name, ic.common_name, \
            ic2.name as parent_name, ic2.common_name as parent_common_name, \
            p.avg as partisan_control \
            from causes_of_death cod \
            inner join icd_10_codes ic on cod.icd_10_codes_id_fkey = ic.id \
            left join icd_10_codes ic2 on ic.parent = ic2.id \
            inner join locations l on cod.locations_id_fkey = l.id \
            inner join (select l.id, AVG(p.control_score) from locations l  \
				inner join partisans p on l.id=p.locations_id_fkey \
				where p.year < 2020 \
				group by l.id ) p on cod.locations_id_fkey =p.id \
            where l.level = 'State'"
    if state != "All":
        sql += "and l.name = '" + state + "'"
    if age != "All":
        sql += " and cod.start_age = '" + str(age) + "'"
    if cod != "All":
        sql += " and ic.common_name = '" + cod + "'"
    if year != "All":
        sql += " and cod.year = '" + str(year) + "'"
        
    con = get_db_connection()
    df = pd.read_sql(sql=text(sql), con = con.engine.connect())
    
    return df


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



def get_age_cod_db(age, all_ages=False, state="Michigan"):
    '''Returns a list of labels, parents, and values for a given state and age group'''
    
    # TODO: move this to YAML
    sql = "select l.name as state, l.acronym, l.fips, \
            cod.rate, cod.deaths, cod.population, cod.start_age, cod.end_age, \
            ic.name, ic.common_name, \
            ic2.name as parent_name, ic2.common_name as parent_common_name \
            from causes_of_death cod \
            inner join icd_10_codes ic on cod.icd_10_codes_id_fkey = ic.id \
            left join icd_10_codes ic2 on ic.parent = ic2.id \
            inner join locations l ON cod.locations_id_fkey = l.id \
            where cod.year=2019 and l.level = 'State' and cod.start_age = '" + str(age) + "' and l.name = '" + state + "'" 
            
    con = get_db_connection()
    
    df = pd.read_sql(sql=text(sql), con = con.engine.connect())
    df = adjust_parent_rate(df)
    
    
    labels = [age]
    parents = [""]
    if all_ages:
        parents = [state]
    values = [0]
    values.extend(list(df["rate"]))
    for i, row in df.iterrows():
        labels.append(row["common_name"]+ ": " + str(row["start_age"]) + "-" + str(row["end_age"]))
        if row["parent_common_name"]!= None:
            parents.append(row["parent_common_name"]+ ": " + str(row["start_age"]) + "-" + str(row["end_age"]))
        else: 
            parents.append(age)
            values[0] += row["rate"]
    return labels, parents, values, df, values[0]

def adjust_parent_rate(df):
    for i in range(5):
        for name in df["parent_name"].unique():
            age = df["start_age"].unique()[0]
            parent_df = df[df["parent_name"] == name]
            parent_rate = parent_df["rate"].sum()
            orignal_rate = 0
            try:
                orignal_rate = df[df["name"] == name]["rate"].values[0]
            except:
                pass
            if parent_rate > orignal_rate:
                df.loc[df["name"] == name, "rate"] = parent_rate
    return df
    

    
def get_all_age_cod(state = "Michigan"):
    '''Returns a list of labels, parents, and values for a given state and all age groups'''
    #df = load_state_cod(state = state)
    ages = [0, 1, 5, 15, 25, 35, 45, 55, 65, 75, 85]
    all_labels = [state]
    all_parents = [""]
    all_values = [0]
    for age in ages:
        labels, parents, values, df_age, total = get_age_cod_db(age, True, state)
        all_labels.extend(labels)
        all_parents.extend(parents)
        all_values.extend(values)
        all_values[0] += total
    return all_labels, all_parents, all_values