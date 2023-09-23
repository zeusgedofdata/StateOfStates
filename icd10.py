import pandas as pd
import json
import numpy as np
import math
import re


class Node: 
    def __init__(self, name, common_name=None):
        self.strip_ranges(name)
        self.strip_chapters(name)
        self.name = name
        self.common_name = self.find_common_name()
        if "#" in name:
            self.base = True
        else:
            self.base = False
        self.children = []
        self.parents = []

    def find_common_name(self):
        '''Find the common name of an ICD-10 code'''
        f = open("data/Life/Deaths/ICD10Translation.json")
        translation = json.load(f)
        try:
            return translation[self.name]
        except:
            return self.name.split("(")[0].strip()
            
    def add_child(self, child):
        '''Add a child to the node'''
        self.children.append(child)
        
    def add_parent(self, parent):
        '''Add a parent to the node'''
        self.parents.append(parent)
        
    def get_range_size(self):
        size = 0
        for chapter in self.ranges.keys():
            size+=len(self.ranges[chapter])
        return size


    def strip_chapters(self, icd_10):
        '''Strip the chapter from an ICD-10 code'''
        codes = icd_10.split("(")[-1].split(",")
        self.chapters = list(set([s[0] for s in codes]))
    
    def strip_ranges(self, icd_10):
        '''Strip the range from an ICD-10 code'''
        codes = re.findall(r'\b[A-Z0-9]+(?:\.[A-Z0-9]+)?(?:-[A-Z0-9]+(?:\.[A-Z0-9]+)?)?\b', icd_10)
        self.ranges = {}
        for code in codes:
            chapter = code[0]
            if chapter not in self.ranges:
                self.ranges[chapter] = []
            self.ranges[chapter].append(re.sub(r'[A-Z]', '', code))
        for chapter in self.ranges.keys():
            self.ranges[chapter] = self.expand_ranges(self.ranges[chapter])
    
    
    
    def expand_ranges(self, input_list):
        '''Expand a list of numbers and ranges into a list of numbers'''
        expanded_list = []

        for item in input_list:
            if '-' in item:
                start, end = item.split('-')
                start_float = float(start) if '.' in start else int(start)
                end_float = float(end.split('.')[0]) if '.' in end else int(end)

                expanded_range = [start_float + i for i in range(int(end_float) - int(start_float) + 1)]
                expanded_list.extend(expanded_range)
            else:
                try:
                    expanded_list.append(math.floor(float(item)))
                except Exception as e:
                    print("Error: " + item)
        return expanded_list
    
    def toJSON(self):
        return {"common_name": self.common_name, "children": [child.toJSON() for child in self.children], "parents": [parent.name for parent in self.parents], "ranges": self.ranges, "chapters": self.chapters, "base": self.base}
    def toFlatJSON(self):
         return {"common_name": self.common_name, "children": [child.name for child in self.children], "parents": [parent.name for parent in self.parents], "ranges": self.ranges, "chapters": self.chapters, "base": self.base}

    
def commonChars(str1, str2):
    # Convert the strings into sets of characters
    str1 = "".join(str1)
    str2 = "".join(str2)    
    set1 = set(str1)
    set2 = set(str2)
     
    # Find the intersection of the sets
    common = set1.intersection(set2)
     
    if len(common) > 0:
        return True
    else:
        return False            

        
def determine_child(node_a, node_b):
    for chapter in node_a.ranges.keys():
        if chapter in node_b.ranges.keys():
            #Determin intersection of the two ranges
            if set(node_a.ranges[chapter]).intersection(set(node_b.ranges[chapter])):
                if len(node_a.ranges[chapter]) < len(node_b.ranges[chapter]):
                    return True
            return False

     
    
df = pd.read_csv(r"data/Life/Deaths/StateDeathsAge.txt", delimiter="	", na_values = ['Not Applicable'])
df = df.dropna(subset=["State","ICD-10 113 Cause List Code", "Population"])
icd_10s = df["ICD-10 113 Cause List"].unique()

all_codes = []
for icd_10 in icd_10s:
    all_codes.append(Node(icd_10))


for node in all_codes:
    for other_node in all_codes:
        if node != other_node:
            try:
                if determine_child(node, other_node):
                    other_node.add_child(node)
                    node.add_parent(other_node)
            except ValueError as e:
                None
                print(e)
                print(node.name)


for node in all_codes:
    if len(node.parents)>1:
        for parent_a in node.parents:
            for parent_b in node.parents:
                if parent_a != parent_b:
                    if parent_a.get_range_size() > parent_b.get_range_size():
                        try:
                            node.parents.remove(parent_a)
                            parent_a.children.remove(node)
                        except Exception as e:
                            print(e)
                            print(node.name + " " + parent_a.name)
print(all_codes[0].toJSON())
json_object = {}
for node in all_codes:
    if node.parents == []:
        json_object[node.name] = node.toJSON()
        
with open("icd-10-Structure.json", "w") as outfile:
    json.dump(json_object, outfile)   
    
json_object = {}


        
for node in all_codes:
        json_object[node.name] = node.toFlatJSON()
        
json_object["#Accidents (unintentional injuries) (V01-X59,Y85-Y86)"]["children"].append("Transport accidents (V01-V99,Y85)")
json_object["#Accidents (unintentional injuries) (V01-X59,Y85-Y86)"]["parents"] = []

json_object["Transport accidents (V01-V99,Y85)"]["parents"] = ["#Accidents (unintentional injuries) (V01-X59,Y85-Y86)"]
json_object["Other land transport accidents (V01,V05-V06,V09.1,V09.3-V09.9,V10-V11,V15-V18,V19.3,V19.8-V19.9,V80.0-V80.2,V80.6-V80.9,V81.2-V81.9,V82.2-V82.9,V87.9,V88.9,V89.1,V89.3,V89.9)"]["parents"] = ["Transport accidents (V01-V99,Y85)"]
json_object["Accidental drowning and submersion (W65-W74)"]["parents"] = ["Nontransport accidents (W00-X59,Y86)"]
json_object["Other and unspecified nontransport accidents and their sequelae (W20-W31,W35-W64,W75-W99,X10-X39,X50-X59,Y86)"]["parents"] = ["Nontransport accidents (W00-X59,Y86)"]


json_object["Other forms of chronic ischemic heart disease (I20,I25)"]["parents"] = ["Ischemic heart diseases (I20-I25)"]
json_object["All other forms of chronic ischemic heart disease (I20,I25.1-I25.9)"]["parents"] = ["Other forms of chronic ischemic heart disease (I20,I25)"]
json_object["Acute myocardial infarction (I21-I22)"]["parents"]= ["Ischemic heart diseases (I20-I25)"]

json_object["All other forms of heart disease (I26-I28,I34-I38,I42-I49,I51)"]["parents"] = ["Other heart diseases (I26-I51)"]
json_object["Heart failure (I50)"]["parents"] = ["Other heart diseases (I26-I51)"]
json_object["Atherosclerotic cardiovascular disease, so described (I25.0)"]["parents"] = ["Ischemic heart diseases (I20-I25)"]
json_object["Other acute ischemic heart diseases (I24)"]["parents"] = ["Ischemic heart diseases (I20-I25)"]
json_object["Acute and subacute endocarditis (I33)"]["parents"] = ["Other heart diseases (I26-I51)"]
json_object["Diseases of pericardium and acute myocarditis (I30-I31,I40)"]["parents"] = ["Other heart diseases (I26-I51)"]


json_object["Accidental exposure to smoke, fire and flames (X00-X09)"]["parents"] = ["Nontransport accidents (W00-X59,Y86)"]
json_object["Accidental poisoning and exposure to noxious substances (X40-X49)"]["parents"] = ["Nontransport accidents (W00-X59,Y86)"]

json_object['Motor vehicle accidents (V02-V04,V09.0,V09.2,V12-V14,V19.0-V19.2,V19.4-V19.6,V20-V79,V80.3-V80.5,V81.0-V81.1,V82.0-V82.1,V83-V86,V87.0-V87.8,V88.0-V88.8,V89.0,V89.2)']["children"] =[]

json_object["Atherosclerotic cardiovascular disease, so described (I25.0)"]["parents"] = ["Other forms of chronic ischemic heart disease (I20,I25)"]


json_object["Assault (homicide) by other and unspecified means and their sequelae (*U01.0-*U01.3,*U01.5-*U01.9,*U02,X85-X92,X96-Y09,Y87.1)"]["parents"] = ["#Assault (homicide) (*U01-*U02,X85-Y09,Y87.1)"]
json_object["Assault (homicide) by other and unspecified means and their sequelae (*U01.0-*U01.3,*U01.5-*U01.9,*U02,X85-X92,X96-Y09,Y87.1)"]["children"] = []
json_object["#Assault (homicide) (*U01-*U02,X85-Y09,Y87.1)"]["parents"] = []

json_object['#Accidents (unintentional injuries) (V01-X59,Y85-Y86)']["children"].remove('Other land transport accidents (V01,V05-V06,V09.1,V09.3-V09.9,V10-V11,V15-V18,V19.3,V19.8-V19.9,V80.0-V80.2,V80.6-V80.9,V81.2-V81.9,V82.2-V82.9,V87.9,V88.9,V89.1,V89.3,V89.9)')
json_object['#Accidents (unintentional injuries) (V01-X59,Y85-Y86)']["children"].remove('Other and unspecified nontransport accidents and their sequelae (W20-W31,W35-W64,W75-W99,X10-X39,X50-X59,Y86)')
json_object['#Diseases of heart (I00-I09,I11,I13,I20-I51)']["children"].remove('Atherosclerotic cardiovascular disease, so described (I25.0)')

json_object["Other and unspecified events of undetermined intent and their sequelae (Y10-Y21,Y25-Y34,Y87.2,Y89.9)"]["children"] =[]


json_object["#Intentional self-harm (suicide) (*U03,X60-X84,Y87.0)"]["parents"]=[]


json_object["Intentional self-harm (suicide) by discharge of firearms (X72-X74)"]["parents"]=["#Intentional self-harm (suicide) (*U03,X60-X84,Y87.0)"]


json_object["Intentional self-harm (suicide) by other and unspecified means and their sequelae (*U03,X60-X71,X75-X84,Y87.0)"]["parents"]=["#Intentional self-harm (suicide) (*U03,X60-X84,Y87.0)"]
json_object["Other and unspecified events of undetermined intent and their sequelae (Y10-Y21,Y25-Y34,Y87.2,Y89.9)"]["children"] =[]

json_object["Ischemic heart diseases (I20-I25)"]["children"].remove("All other forms of chronic ischemic heart disease (I20,I25.1-I25.9)")
json_object["#Diseases of heart (I00-I09,I11,I13,I20-I51)"]["children"].remove("All other forms of chronic ischemic heart disease (I20,I25.1-I25.9)")

#"Ischemic heart diseases (I20-I25)"


#Accidents (unintentional injuries) (V01-X59,Y85-Y86)
for disease in json_object.keys():
    if len(json_object[disease]["parents"]) > 0:
        json_object[json_object[disease]["parents"][0]]["children"].append(disease)
        json_object[json_object[disease]["parents"][0]]["children"] = list(set(json_object[json_object[disease]["parents"][0]]["children"]))
    if len(json_object[disease]["children"]) > 0:
        disease_children = json_object[disease]["children"]
        for child in json_object[disease]["children"]:
            grand_children = json_object[child]["children"]
            common_members = list(set.intersection(set(disease_children), set(grand_children)))
            for common_member in common_members:
                json_object[disease]["children"].remove(common_member)
                


#Diseases of pericardium and acute myocarditis (I30-I31,I40)
#Atherosclerotic cardiovascular disease, so described (I25.0)
#Diseases of heart (I00-I09,I11,I13,I20-I51)


#json_object["Transport accidents (V01-V99,Y85)"]["parents"] = ["#Accidents (unintentional injuries) (V01-X59,Y85-Y86)"]
        
        
with open("icd-10-flat-Structure.json", "w") as outfile:
    json.dump(json_object, outfile)                      

        #The parent is the one with the smallest list? 

