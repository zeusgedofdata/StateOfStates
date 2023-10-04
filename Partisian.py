import pandas as pd
import json
import numpy as np
import addfips


def get_part_cat(score):
    if score<-.3:
        return "Left"
    elif score<.3:
        return "Center"
    else:
        return "Right"
    
def get_state_part_score_all():
    af = addfips.AddFIPS()
    us_state_to_abbrev = {"Alabama": "AL","Alaska": "AK","Arizona": "AZ","Arkansas": "AR","California": "CA","Colorado": "CO","Connecticut": "CT","Delaware": "DE","Florida": "FL","Georgia": "GA","Hawaii": "HI","Idaho": "ID","Illinois": "IL","Indiana": "IN","Iowa": "IA","Kansas": "KS","Kentucky": "KY","Louisiana": "LA","Maine": "ME","Maryland": "MD","Massachusetts": "MA","Michigan": "MI","Minnesota": "MN","Mississippi": "MS","Missouri": "MO","Montana": "MT","Nebraska": "NE","Nevada": "NV","NewHampshire": "NH","NewJersey": "NJ","NewMexico": "NM","NewYork": "NY","NorthCarolina": "NC","NorthDakota": "ND","Ohio": "OH","Oklahoma": "OK","Oregon": "OR","Pennsylvania": "PA","RhodeIsland": "RI","SouthCarolina": "SC","SouthDakota": "SD","Tennessee": "TN","Texas": "TX","Utah": "UT","Vermont": "VT","Virginia": "VA","Washington": "WA","WestVirginia": "WV","Wisconsin": "WI","Wyoming": "WY","District of Columbia": "DC","American Samoa": "AS","Guam": "GU","Northern Mariana Islands": "MP","Puerto Rico": "PR","United States Minor Outlying Islands": "UM","U.S. Virgin Islands": "VI"}
    df = pd.read_csv(r"D:\StateOfStates\data\Partisian\StateControl.txt", delimiter=" ")
    df["Part_Score"]  = df.StateControl.map({"Rep":1,"Divided":0,"Dem":-1, np.nan:np.nan})
    df["State"] = df.STATE.map(us_state_to_abbrev)
    df["fips"] = df["State"].apply(lambda x: af.get_state_fips(x))
    return df
    

def get_state_part_score(year=2024):
    af = addfips.AddFIPS()
    us_state_to_abbrev = {"Alabama": "AL","Alaska": "AK","Arizona": "AZ","Arkansas": "AR","California": "CA","Colorado": "CO","Connecticut": "CT","Delaware": "DE","Florida": "FL","Georgia": "GA","Hawaii": "HI","Idaho": "ID","Illinois": "IL","Indiana": "IN","Iowa": "IA","Kansas": "KS","Kentucky": "KY","Louisiana": "LA","Maine": "ME","Maryland": "MD","Massachusetts": "MA","Michigan": "MI","Minnesota": "MN","Mississippi": "MS","Missouri": "MO","Montana": "MT","Nebraska": "NE","Nevada": "NV","NewHampshire": "NH","NewJersey": "NJ","NewMexico": "NM","NewYork": "NY","NorthCarolina": "NC","NorthDakota": "ND","Ohio": "OH","Oklahoma": "OK","Oregon": "OR","Pennsylvania": "PA","RhodeIsland": "RI","SouthCarolina": "SC","SouthDakota": "SD","Tennessee": "TN","Texas": "TX","Utah": "UT","Vermont": "VT","Virginia": "VA","Washington": "WA","WestVirginia": "WV","Wisconsin": "WI","Wyoming": "WY","District of Columbia": "DC","American Samoa": "AS","Guam": "GU","Northern Mariana Islands": "MP","Puerto Rico": "PR","United States Minor Outlying Islands": "UM","U.S. Virgin Islands": "VI"}
    df = pd.read_csv(r"D:\StateOfStates\data\Partisian\StateControl.txt", delimiter=" ")
    df["Part_Score"]  = df.StateControl.map({"Rep":1,"Divided":0,"Dem":-1, np.nan:np.nan})
    df["State"] = df.STATE.map(us_state_to_abbrev)
    df["fips"] = df["State"].apply(lambda x: af.get_state_fips(x))
    df = df[df.Year < year]
    state_score = df.groupby(by=["State", "fips"]).Part_Score.mean().sort_values().reset_index()
    state_score["Year"] = year
    state_score = state_score.dropna()
    state_score["category"] = state_score["Part_Score"].apply(lambda x:get_part_cat(x))
    return state_score

def get_county_votes(year = 2020, avg = False):
    df = pd.read_csv(f"D:\StateOfStates\data\Partisian\countypres_2000-2020.csv")
    df["percent"] = df["candidatevotes"]/df["totalvotes"]
    df["fips"] = df.county_fips.apply(lambda x: str(x).split(".")[0].zfill(5))
    df = df[df["mode"]=="TOTAL"]
    
    if avg: 
        df = df[df.year <= year]
        df = df.groupby(by=["state", "county_name", "county_fips"]).mean().reset_index()
    
    return df