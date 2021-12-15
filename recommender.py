# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

import plotly.express as px
import re

products = pd.read_csv('final_product.csv')


# Recommend for product now
def recommender(view_product, dictionary, tfidf, index):
    # Convert search words into Sparse Vectors
    view_product = view_product.lower().split()
    kw_vector = dictionary.doc2bow(view_product)
    # print("View product's vector:")
    # print(kw_vector)
    # Similarity calculation
    sim = index[tfidf[kw_vector]]

    # print result
    list_id = []
    list_score = []
    for i in range(len(sim)):
        list_id.append(i)
        list_score.append(sim[i])

    df_result = pd.DataFrame({'id': list_id,
                                'score': list_score})

    # Five highest scores
    five_highest_score = df_result.sort_values(by='score',ascending=False).head(6)
    # print("Five highest score:")
    # print(five_highest_score)
    # print("Ids to list:")
    idToList = list(five_highest_score['id'])
    # print(idToList)

    products_find = products[products.index.isin(idToList)]
    results = products_find[['index','item_id','name']]
    results = pd.concat([results,five_highest_score],axis=1).sort_values(by='score',ascending=False)
    return results
    