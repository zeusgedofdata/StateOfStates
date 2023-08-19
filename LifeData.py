import json
import pandas as pd
import numpy as np

class LifeData():
    def __init__(self):
        file = open('./cdc_data.json')
        self.meta_files = json.load(file)["CDC Files"]
        self.data_mapping = {}
        dfs = []
        for meta_file in self.meta_files:
            df = self.clean_data(meta_file)
            self.data_mapping[meta_file['Indicator Name']] = df.copy()
            dfs.append(df)
        self.all_data = pd.concat(dfs)
            
    def get_single_indicator(self, indicator, most_recent = True):
        df = self.data_mapping[indicator]
        if most_recent:
            df = self.filter_data_most_recent(df)
        return df
    
    
    def get_state_card_data(self, state):
        df = self.all_data[self.all_data["State"] == state]
        df.reset_index()
        df = self.filter_data_most_recent(df)
        return df[["State","Year","Indicator","Rate", "Rank"]]
    
    def get_long_format(self):
        df = self.filter_data_most_recent(self.all_data)
        pivot_df= df.pivot(index="State", columns="Indicator", values="Rate")
        pivot_df = pivot_df.reset_index()
        return pivot_df

        
    def clean_data(self, meta_file):
        df = pd.read_csv(meta_file['File Location'])
        df["Indicator"] = meta_file['Indicator Name']
        df = df.rename(columns={"YEAR":"Year","STATE":"State", meta_file['Rate Name']:"Rate"})
        if meta_file['Missing Data']:
            df["Rate"] = df["Rate"].replace(meta_file['Missing Data'], np.NaN)
            df.dropna(subset=["Rate"], inplace=True)
        df["Rate"] = df["Rate"].astype(float)
        df["Rate"] = df["Rate"].round(2)
        df = df[["State", "Year", "Indicator", "Rate"]]
        df = self.get_rank_year(df)
        return df
    
    
    def get_rank_year(self, df):
        dfs = []
        for year in df.Year.unique():
            year_df = df[df.Year == year].copy()
            year_df["Rank"] = year_df.Rate.rank().astype(int)
            #print(f"Checking for NAs in {year} ind {year_df['Indicator'].unique()} rate {year_df['Rate'].isna().sum()}")
            #print("Checking for NAs")
            #print(year[year["Rank"].isna()])
            #print(year_df["Rate"].isna().sum())
            life_exp = year_df[year_df["Indicator"]=="Life Exp"].copy()
            life_exp["Rank"] = life_exp.Rate.rank(ascending=False).astype(int)
            year_df.loc[life_exp.index] = life_exp
            dfs.append(year_df)
        df = pd.concat(dfs)
        df["Rank"] = df.Rank.astype(int)
        return df
    

        

    def filter_data_most_recent(self, df):
        df = df.reset_index()
        idxs = []
        for ind in df.Indicator.unique():
            ind_df = df[df.Indicator==ind]
            ind_df = ind_df[ind_df.Year == ind_df.Year.max()]
            idxs.extend(list(ind_df.index))

        df = df.loc[idxs]
        return df

        
        
    