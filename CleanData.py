import pandas as pd
import json
import numpy as np

def clean_state_cod(year=2019):
    '''Returns a dataframe of cause of death data for a single state broken up by age group'''
    df_heirarchy = get_heirarchy_df()
    #rotate Matrix
    df_heirarchy = df_heirarchy.transpose()
    df_heirarchy.reset_index(inplace=True)
    df_heirarchy.rename(columns={"index":"ICD-10 113 Cause List"}, inplace=True)
    df_heirarchy

    df = pd.read_csv(r"D:/StateOfStates/data/Life/Deaths/StateAgeCod/StateAgeCOD"+str(year)+".txt", delimiter="	", na_values = ['Not Applicable'])
    df = df.dropna(subset=["State","ICD-10 113 Cause List Code", "Population"]) 
    
    #state_df = state_df[state_df["Ten-Year Age Groups Code"] ==age]
    df["Rate"] = (df["Deaths"] / df["Population"])*100000
    df["Rate"] = np.round(df["Rate"], 2)

    df_final = pd.merge(df, df_heirarchy, on="ICD-10 113 Cause List")
    df_final = df_final.drop(columns=["ICD-10 113 Cause List Code", "ranges", "chapters", "base", "Notes"])
    return df_final

def get_heirarchy_df():
    '''Returns a dataframe of the ICD-10 heirarchy'''
    flat_heirarchy = json.load(open(r"D:\StateOfStates\icd-10-flat-Structure.json"))
    df_heirarchy = pd.DataFrame(flat_heirarchy)
    return df_heirarchy


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