#!/usr/bin/env python
# coding: utf-8

# In[19]:


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


# In[20]:


tips = sns.load_dataset("tips")


# In[21]:


tips


# In[22]:


#scatter_plot = axis level function
sns.scatterplot(data = tips, x = "total_bill", y = "tip")


# In[59]:


sns.relplot(data=tips, x='total_bill', y='tip', kind='scatter',hue='sex',style='time',size='size')


# In[23]:


sns.relplot(data = tips, x = 'total_bill', y = "tip", kind = "scatter" )


# In[60]:


#lione plot
gap = px.data.gapminder()
gap


# In[61]:


temp_df = gap[gap["country"] == "India"]
temp_df


# In[62]:


temp_df = gap[gap["country"].isin(["India", "Pakistan", "China"])]
temp_df


# In[63]:


#axis level function
sns.lineplot(data = temp_df, x = "year" , y = "lifeExp")


# In[71]:


#same graph using rel plot
sns.relplot(kind = "scatter", data = temp_df, x = "year", y = "lifeExp", hue = "country")


# In[67]:


sns.relplot(data = tips, x = "total_bill", y = "tip", kind = "scatter", hue = "sex")


# In[72]:


sns.relplot(kind = "scatter", data = tips, x = "total_bill", y = "tip", col = "sex")


# In[73]:


sns.relplot(kind = "scatter", data = tips, x = "total_bill", y = "tip", col = "sex", row = "smoker")


# In[74]:


sns.relplot(kind = "scatter", data = tips, x = "total_bill", y = "tip", col = "sex", row = "day")


# In[77]:


import pandas as pd
tips['total_bill'] = pd.to_numeric(tips['total_bill'], errors='coerce')
tips['tip'] = pd.to_numeric(tips['tip'], errors='coerce')
tips['sex'] = tips['sex'].astype('category')
tips['day'] = tips['day'].astype('category')
tips.dropna(subset=['total_bill', 'tip', 'sex', 'day'])


# In[80]:


sns.relplot(data=gap, x='lifeExp', y='gdpPercap', kind='scatter', col='year')


# In[81]:


#usage of col_wrap to increase readability
sns.relplot(data=gap, x='lifeExp', y='gdpPercap', kind='scatter', col='year', col_wrap=3)


# DISTRIBUTION PLOTS
# 1. used for univariate analysis
# 2. used to find out the distribution
# 3. Range of the observation
# 4. Central Tendency
# 5. is the data bimodal?
# 6. Are there outliers?
# 7. Plots under distribution plot
# 
# TYPES:
# 
# 1. histplot
# 2. kdeplot
# 3. rugplot

# In[82]:


# figure level -> displot
# axes level -> histplot -> kdeplot -> rugplot


# In[86]:


#plotting univariate histogram
sns.histplot(data = tips, x = "total_bill")


# In[90]:


sns.histplot(tips["total_bill"], bins = 20)


# In[105]:


#figure level function
sns.distplot(tips["total_bill"], hist = True)
#doesn't work anymore


# In[92]:


sns.histplot(data = tips, x = "day")


# In[93]:


#hue parameter
sns.histplot(data = tips, x = "day", hue = "sex")


# In[95]:


titanic = sns.load_dataset("titanic")


# In[96]:


titanic


# In[97]:


sns.histplot(data = titanic, x = "age")


# In[98]:


sns.histplot(data = titanic, x = "age", hue = "sex")


# KDE PLOT

# In[99]:


sns.kdeplot(data = tips, x = "total_bill")


# In[107]:


sns.distplot(tips['total_bill'], hist=False, kde=True)


# In[116]:


sns.kdeplot(data = tips, x = 'total_bill', hue='sex')


# In[118]:


# Bivariate histogram
# A bivariate histogram bins the data within rectangles that tile the plot 
# and then shows the count of observations within each rectangle with the fill color

# sns.histplot(data=tips, x='total_bill', y='tip')
sns.displot(data=tips, x='total_bill', y='tip',kind='hist')


# In[117]:


# Bivariate Kdeplot
# a bivariate KDE plot smoothes the (x, y) observations with a 2D Gaussian
sns.kdeplot(data=tips, x='total_bill', y='tip')


# MATRIX PLOT
# 
# 1. HEAT MAP
# 2. CLUSTER MAP

# In[119]:


# Heatmap

# Plot rectangular data as a color-encoded matrix
temp_df = gap.pivot(index='country',columns='year',values='lifeExp')

# axes level function
plt.figure(figsize=(15,15))
sns.heatmap(temp_df)


# In[120]:


iris = sns.load_dataset("iris")


# Categorical Plots
# 
# Categorical Scatter Plot
# 1. Stripplot
# 2. Swarmplot
# 
# Categorical Distribution Plots
# 1. Boxplot
# 2. Violinplot
# 
# Categorical Estimate Plot -> for central tendency
# 1. Barplot
# 2. Pointplot
# 3. Countplot
# 
# Figure level function -> catplot

# In[122]:


#strip plot
sns.stripplot(data = tips, x = "day", y = "total_bill")


# In[123]:


sns.catplot(data = tips, x = "day", y = "total_bill", kind = "strip")


# In[124]:


sns.catplot(data = tips, x = "day", y = "total_bill", hue = "sex", kind = "strip")


# In[125]:


#swarm plot
sns.catplot(data = tips, x = "day", y = "total_bill", kind = "swarm")


# In[126]:


sns.swarmplot(data = tips, x = "day", y = "total_bill", hue = "sex" )


# In[127]:


#boxplot
sns.catplot(data = tips, x = "day", y = "total_bill", kind = "box" )


# In[128]:


sns.boxplot(data = tips, y = "total_bill")


# In[129]:


#violin plot
sns.violinplot(data = tips, x = "day", y = "total_bill")


# In[131]:


sns.catplot(data = tips, x = "day", y = "total_bill", kind = "violin", hue = "sex", split = True)


# In[132]:


# barplot
import numpy as np
sns.barplot(data=tips, x='sex', y='total_bill')


# In[133]:


sns.barplot(data=tips, x='sex', y='total_bill',estimator = np.min )


# In[134]:


sns.barplot(data=tips, x='sex', y='total_bill', hue = "smoker", estimator = np.min)


# In[135]:


#pointplot
sns.pointplot(data=tips, x='sex', y='total_bill')


# In[137]:


#countplot
sns.countplot(data=tips, x='sex')


# In[138]:


sns.countplot(data=tips, x='sex', hue = "day")


# In[139]:


#faceting using catplot
sns.catplot(data = tips, x = "sex", y = "total_bill", col = "smoker", kind = "box", row = "time")


# REG PLOT

# In[140]:


#axes level
sns.regplot(data = tips, x = "total_bill", y = "tip")


# In[141]:


sns.lmplot(data=tips,x='total_bill',y='tip',hue='sex')


# In[142]:


#residual plot
sns.residplot(data = tips, x = "total_bill", y = "tip")


# PAIRGRID VS PAIRPLOT

# In[143]:


#pairplot
sns.pairplot(iris)


# In[148]:


g = sns.PairGrid(iris)
g.map(sns.scatterplot)


# In[149]:


g = sns.PairGrid(iris)
g.map(sns.histplot)


# In[150]:


# map_diag -> map_upper -> map_lower
g = sns.PairGrid(data=iris,hue='species')
g.map_diag(sns.histplot)
g.map_upper(sns.kdeplot)
g.map_lower(sns.scatterplot)


# In[151]:


# vars
g = sns.PairGrid(data=iris,hue='species',vars=['sepal_width','petal_width'])
g.map_diag(sns.histplot)
g.map_upper(sns.kdeplot)
g.map_lower(sns.scatterplot)


# In[152]:


g = sns.PairGrid(data=iris)
g.map(sns.boxplot)


# JOINTGRID VS JOINTPLOT

# In[154]:


sns.jointplot(data=tips,x='total_bill',y='tip',kind='hist',hue='sex')


# In[155]:


g = sns.JointGrid(data=tips,x='total_bill',y='tip')
g.plot(sns.kdeplot,sns.violinplot)


# In[ ]:




