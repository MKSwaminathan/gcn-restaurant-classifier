# -*- coding: utf-8 -*-
"""EE461P_Misc.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mcQlUV9i_bSEl7MnadNSpJHreL-Bi7s_
"""

import json

#from google.colab import files
#uploaded= files.upload()
#files.download("outfile.csv")

#from google.colab import drive
#drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd drive/My\ Drive/Yelp

import sys
#sys.path.append('../yelp_dataset/')

#f = open('yelp_academic_dataset_business.json',)
#dict = json.load(f)

#Read json entries into a list. Now can basically use the list as a python dictionary
restaurants = []
for line in open('../yelp_dataset/yelp_academic_dataset_business.json', 'r'):
    restaurants.append(json.loads(line))

#Function for computing distance from latitude and longitude coordinates
from math import radians, cos, sin, asin, sqrt 
def distance(lat1, lat2, lon1, lon2): 
      
    # The math module contains a function named 
    # radians which converts from degrees to radians. 
    lon1 = radians(lon1) 
    lon2 = radians(lon2) 
    lat1 = radians(lat1) 
    lat2 = radians(lat2) 
       
    # Haversine formula  
    dlon = lon2 - lon1  
    dlat = lat2 - lat1 
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
  
    c = 2 * asin(sqrt(a))  
     
    # Radius of earth in kilometers. Use 3956 for miles 
    r = 6371
       
    # calculate the result 
    return(c * r)

#Function for calculating the adjacency matrix given the dictionary of restaurants in the city and the number of restaurants in that city
import numpy as np
def calcAdjMatrix(city, num_restaurants):
  dist = np.zeros((num_restaurants,num_restaurants))
  for i in range(num_restaurants):
    for j in range(num_restaurants):
      if (j != i):
        lat1 = city[i]['latitude']
        lat2 = city[j]['latitude']
        lon1 = city[i]['longitude']
        lon2 = city[j]['longitude']
        dist[i][j] = distance(lat1, lat2, lon1, lon2)
  return dist

#Find the cities in the dataset and how many of those cities' restaurants are in the dataset in order
from collections import defaultdict
from collections import OrderedDict
cities = []
for city in restaurants:
    cities.append(city['city'])

#unique list of names
unique_cities = list(set(cities))
#print(unique_cities)

city_dict = defaultdict(int)
for city in cities:
    city_dict[city] += 1
#print(city_dict)
city_dict_descending = OrderedDict(sorted(city_dict.items(), key=lambda kv: kv[1], reverse=True))
#print(city_dict_descending)
city_dict_descending['Pittsburgh']

#Find the states in the dataset and how many of those states' restaurants are in the dataset in order
states = []
for state in restaurants:
    states.append(state['state'])

unique_states = list(set(states))
#print(unique_states)

state_dict = defaultdict(int)
for state in states:
    state_dict[state] += 1
#print(state_dict)
state_dict_descending = OrderedDict(sorted(state_dict.items(), key=lambda kv: kv[1], reverse=True))
#print(state_dict_descending)

#Populating a list for restaurants in a given city.
las_vegas = []
for city in range(len(restaurants)):
  if restaurants[city]['city'] == 'Las Vegas':
    las_vegas.append(restaurants[city])

#Populating a list for restaurants in Pittsburgh
pittsburgh = []
for city in range(len(restaurants)):
  if restaurants[city]['city'] == 'Pittsburgh':
    pittsburgh.append(restaurants[city])

adjMat = calcAdjMatrix(pittsburgh, city_dict_descending['Pittsburgh'])

#adjMat

#For outputting the adjacency matrix to a csv file. Should appear in Drive folder if drive is mounted. Can also download to local drive
import pandas as pd
df = pd.DataFrame(data=adjMat.astype(float))
df.to_csv('pittsburgh.csv', sep=' ', header=False, float_format='%.2f', index=False)

from scipy.sparse import csr_matrix
csr_mat = csr_matrix(adjMat)

#csr_mat

try:
   import cPickle as pickle
except:
   import pickle

with open('test_sparse_array.dat', 'wb') as outfile:
    pickle.dump(csr_mat, outfile, pickle.DEFAULT_PROTOCOL)

#from google.colab import files
#files.download("test_sparse_array.dat")

#print(restaurants[2000])

#Flattening a nested JSON
#!pip install flatten_json

#https://stackoverflow.com/questions/58442723/how-to-flatten-nested-json-recursively-with-flatten-json
from flatten_json import flatten
flattened_restaurants = []
for restaurant in restaurants:
  flattened_restaurants.append(flatten(restaurant))
#print(flattened_restaurants[2000]['attributes_BusinessParking'])

import pandas as pd
from flatten_json import flatten_json
df = pd.DataFrame([flatten_json(x) for x in restaurants])

#df.to_csv('flattened.csv')
#df.shape

#flattened_pittsburgh = []
#for restaurant in pittsburgh:
#  flattened_pittsburgh.append(flatten(restaurant))
df = pd.DataFrame([flatten_json(x) for x in pittsburgh])

#df.head()

#Nan count for each row:
#for i in range(len(df.index)) :
#    print("Nan in row ", i , " : " ,  df.iloc[i].isnull().sum())

#Nan count for each column:
#df.isnull().sum()
#Nan percentage for each column:
#df.isnull().mean()

#Adding binary success label based on whether star count > 3
df['label'] = df['stars'] > 3
df['label'] = df['label'].astype(int)

#df['label']

#remove columns which have greater than threshold NaN values
threshold = 0.3
df1 = df[df.columns[df.isnull().mean() < threshold]]

#df1

df1.to_pickle('flattened_data.dat')

