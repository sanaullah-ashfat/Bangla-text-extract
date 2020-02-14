# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 21:06:07 2020

@author: Sanaullah
"""


import pandas as pd
import numpy as np
import os 
import spacy
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher


df = pd.read_csv("G:/Tasks/Task1/names_sample.csv")

dff = df.fillna(method='bfill')

df1=pd.read_csv("G:/Tasks/Task1/user_raw_data.csv")
df2 =df1.fillna(method='bfill')
house=dff['BusinessName'].values.tolist()


#data = df1.address.iloc[5]
data ='18 GHA Al Madina bhaban banglalink digital communications limited'

tokenizer = spacy.blank('en')


data_show = tokenizer(data)

matcher = PhraseMatcher(tokenizer.vocab, attr='LOWER')

nl = [tokenizer(term)for term in house]

matcher.add("Matching", None, *nl)
textdoc =tokenizer(data)

matches = matcher(textdoc)

for match in matches:
    print(f"Token number {match[1]}: {data_show[match[1]:match[2]]}")









