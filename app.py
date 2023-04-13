#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:25:03 2023

@author: juancarlosmac
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame(
    np.random.randn(200, 2) / [50, 50] + [20.76, -103.4],
    columns=['lat', 'lon'])
Datos = df
# KMeans está en el paquete sklearn.cluster
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=42) # K-Means con K=5
kmeans.fit(Datos) # haz 5 grupos con nuestros datos

#Clusters = pd.DataFrame()
Datos['Kmeans_Clusters'] = kmeans.labels_
print(Datos.head())

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('lat',fontsize=15)
ax.set_ylabel('lon',fontsize=15)
ax.set_title('Grafica de Clusters',fontsize=20)
color_theme = np.array(['blue','green','orange','black','red','cian','skyblue','beige','gold','pink'])
ax.scatter(x = Datos.lat,y=Datos.lon,c=color_theme[Datos.Kmeans_Clusters],s=50)

plt.show()