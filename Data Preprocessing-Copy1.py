#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy,pandas
df = pandas.read_csv("/Users/abby-keisha/Downloads/Demo Datasets/Lesson 3/SalaryGender.csv")
salary = numpy.array(df['Salary'])
gender = numpy.array(df['Gender'])
phd = numpy.array(df['PhD'])
age = numpy.array(df['Age'])
print (df)


# In[66]:


import numpy,pandas

df = pandas.read_csv("/Users/abby-keisha/Downloads/Demo Datasets/Lesson 3/SalaryGender.csv",delimiter = ',')
salary = numpy.array(df['Salary'])
gender = numpy.array(df['Gender'])
phd = numpy.array(df['PhD'])
age = numpy.array(df['Age'])
print (df)
type(df)
df.iloc[:,1:3]
df.mean()
df.median()
df.mode(axis=0)


# In[36]:


import pandas as pd

df = pd.read_csv("/Users/abby-keisha/Downloads/california_housing_train.csv")

print (df)


# In[68]:


import matplotlib.pyplot as plt
import seaborn as sns
correlations = df.corr()
sns.heatmap(data = correlations,square = True, cmap = "bwr")
plt.yticks(rotation=0)
plt.xticks(rotation=90)


# In[69]:


import pandas as pd
df = pd.readf = pd.read_csv('/Users/abby-keisha/Downloads/Demo Datasets/Lesson 3/middle_tn_schools.csv')
df.describe()


# In[71]:


df[['reduced_lunch','school_rating']].groupby(['school_rating']).describe().unstack()


# In[72]:


df[['reduced_lunch','school_rating']].corr()


# In[76]:


df.isna().any()


# In[106]:


import seaborn as sns
sns.boxplot(x=df['reduced_lunch'])


# In[107]:


import seaborn as sns
sns.boxplot(x=df['school_rating'])


# In[113]:


filter=df['school_rating'].values>1
df_outlier_rem=df[filter]
df_outlier_rem


# In[117]:


filter=df['reduced_lunch'].values>50
df_outlier_rem=df[filter]
df_outlier_rem


# In[118]:


from sklearn.datasets import load_diabetes


# In[119]:


dataset = load_diabetes()
dataset


# In[120]:


dataset.target


# In[121]:


dataset['feature_names']


# In[125]:


import pandas as pd
import numpy as np


# In[127]:


df = pd.DataFrame(data=np.c_[dataset['data'],dataset['target']],columns = dataset['feature_names']+ ['target'])
df


# In[128]:


df.isnull().any()


# In[131]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

for column in df:
    plt.figure()
    df.boxplot([column])


# In[132]:


import pandas as pd
import numpy as np
df=pd.Series(np.arange(1,51))
print(df.head(6))


# In[133]:


import pandas as pd
import numpy as np
df=pd.Series(np.arange(1,51))
print(df.tail(6))


# In[135]:


import pandas as pd
import numpy as np
df=pd.Series(np.arange(1,51))
print(df.values)


# In[139]:


import pandas as pd
world_cup={'Team':['West Indies','Westindies','India','Australia','Pakistan','SriLanka','Australia','Australia','Australia','Insia','Australia'],'Rank':[7,7,2,1,6,4,1,1,1,2,1],'Year':[1975,1979,1983,1987,1992,1996,1999,2003,2007,2011,2015]}
df=pd.DataFrame(world_cup)
print(df.groupby(['Team','Rank’]).groups)


# In[141]:


import pandas as pd
world_cup={'Team':['West Indies','West Indies','India','Australia','Pakistan','Sri Lanka','Australia','Australia','Australia',' India','Australia'],'Rank':[7,7,2,1,6,4,1,1,1,2,1],'Year':[1975,1979,1983,1987,1992,1996,1999,2003
,2007,2011,2015]}
df=pd.DataFrame(world_cup)

print(df.groupby(['Team','Rank']).groups)                


# In[143]:


import pandas
world_champions={'Team':['India','Australia','West Indies','Pakistan','Sri Lanka'],'Icc_Rank':[2,3,7,8,4],'World_champions_Year':[2011,20115,1979,1992,1996],'Points':[874,787,753,673,855]}

chokers={'Team':['South Africa','New Zealand','Zimbabwe'],'ICC_rank':[1,5,9],'Points':[895,764,656]}     
df1=pandas.DataFrame(world_champions)
df2=pandas.DataFrame(chokers)
print(pandas.concat([df1,df2],axis=1))                         


# In[144]:


import pandas
champion_stats={'Team':['India','Australia','West Indies','Pakistan','Sri Lanka'],'Icc_Rank':[2,3,7,8,4],'World_champions_Year':[2011,2015,1979,1992,1996],'Points':[874,787,753,677,853]}

match_stats={'Team':['India','Australia','West Indies','Pakistan','Sri Lanka'],'World_cup_played':[11,10,11,9,8],'OdIs_played':[733,988,712,679,662]}

df1=pandas.DataFrame(champion_stats)
df2=pandas.DataFrame(match_stats)
print(df1)
print(df2)
print(pandas.merge(df1,df2,on='Team'))


# In[152]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[153]:


north_america = pd.read_csv('/Users/abby-keisha/Downloads/Demo Datasets/Lesson 3/north_america_2000_2010.csv',index_col = 0)
south_america = pd.read_csv('/Users/abby-keisha/Downloads/Demo Datasets/Lesson 3/south_america_2000_2010.csv',index_col = 0)


# In[158]:


north_america


# In[160]:


south_america


# In[161]:


americas = pd.concat([north_america,south_america])
americas


# In[37]:



df = pandas.read_csv("/Users/abby-keisha/Downloads/california_housing_train.csv")

print (df)

