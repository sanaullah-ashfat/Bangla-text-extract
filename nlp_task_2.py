# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 01:10:01 2020

@author: Sanaullah
"""



import pandas as pd
import numpy as np
import os 
import spacy
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher
import re

df1=pd.read_csv("G:/Tasks/Task2/user_raw_data.csv")

df2 =df1.fillna(method='bfill')
df3=pd.read_csv("G:/Tasks/Task2/roads.csv")
df3 =df3.fillna(method='bfill')
road=df3['Road'].values.tolist()

#data = df1.address.iloc[3]
data = 'hose no 12, badda link Road dhaka'

tokenizer = spacy.blank('en')

data_show = tokenizer(data)

matcher = PhraseMatcher(tokenizer.vocab, attr='LOWER')

menu_tokens_list = [road for item in road]

nl = [tokenizer(term)for term in road]

matcher.add("Matching", None, *nl)
textdoc =tokenizer(data)

matches = matcher(textdoc)


for match in matches:
    print(f"Token number {match[1]}: {data_show[match[1]:match[2]]}")


