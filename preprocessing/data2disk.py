
import pandas as pd
import os

SPINKLED_SUB_KEY = 'sprinkled_subs'
CATEGORIES_KEY = 'categories'

def save_data(p,sprinkled_subs,categories):
    df = pd.DataFrame({
        SPINKLED_SUB_KEY:sprinkled_subs,
        CATEGORIES_KEY:categories
    })
    if not '.csv' in p:
        path = p+'.csv'    
    df.to_csv(path, sep='\t')

def load_data(p):
    if not '.csv' in p:
        path = p+'.csv'
    if not os.path.exists(path):
        return None,None
    df = pd.read_csv(path, sep='\t', header=None)
    #File has three columns: id,sprinkled_subs,categories
    #Get sprinkled subs on second column and remove header
    return df[1][1:].values, [n.replace("'","")[1:-1].split(",") for n in df[2][1:].values]