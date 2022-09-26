#import libraries
import pandas as pd
from pandas import DataFrame
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import make_blobs


print ('M3 Activity 2')
print ('Boado, John David')
print ('Bulaon, Madeleine')
print ('Cabrera, Marc Ivan')
print ('Ebora, John Rhanzel')


#dataset
df_covid = pd.read_csv(r'C:\Users\User\Desktop\School Thing\3rd year\1st Sem\ITEL1 Professional Elective for IT Majors 1\Prog\Covid2020.csv')

# Identify your data type using df_alumni.dtypes command.
# With its given output you can now analyze the types of data and its attributes.
df_covid.dtypes

# View the DataFrame using the df_alumni.head()
# The default is 5 rows (0-4) but you can increase it by adding specific values in () 
df_covid.head()

# View the last Rows of DataFrame using the df_alumni.head()
# The default is 5 rows (0-4) but you can increase it by adding specific values in () 
df_covid.tail()

#A. Delete
# identify the required attributes in the dataset using df_alumni.columns command.
# Based on its data frame and the column index as show in the given output, 
# you can select what columns you want to remove. 
df_covid.columns

# Delete unnecessary data.
# By analyzing the  columns you can immediately delete unnecessary attributes in the dataset.
del df_covid['CaseCode']
del df_covid['DateRepConf']
del df_covid['DateDied']
del df_covid['DateRecover']
del df_covid['DateRepRem']
del df_covid['ProvRes']
del df_covid['CityMunRes']
del df_covid['CityMuniPSGC']
del df_covid['DateOnset']


# Review columns to see if it is sucessfully deleted.
df_covid.columns

#B. Fitting dataset
#Age Group, Sex, Removal Type, Admitted, Quarantined, Health Status, Pregnant 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df_covid['Sex'] = le.fit_transform(df_covid['Sex'])
df_covid['RemovalType'] = le.fit_transform(df_covid['RemovalType'])

# View the dataset head to see the output of new encoded value
df_covid.head()

# re-encode individually other data which "did not automatically transformed" with unique command
df_covid['Admitted'] = le.fit_transform(df_covid['Admitted'])
df_covid['Admitted'].unique()

# re-encode individually other data which "did not automatically transformed" with unique command
df_covid['Quarantined'] = le.fit_transform(df_covid['Quarantined'])
df_covid['Quarantined'].unique()

# re-encode individually other data which "did not automatically transformed" with unique command
df_covid['Pregnanttab'] = le.fit_transform(df_covid['Pregnanttab'])
df_covid['Pregnanttab'].unique()

# There are types of data that need to transform or re-encode into specific value. 
# Like in this case, the field_of_work, Length of Service and estd-monthly_income.

#Age group
#0  = toddler 
#1 = children 
#2 = pre-teens
#3 = teenager
#4 = adult 
#5 = senior

df_covid['AgeGroup'] = np.where(df_covid['AgeGroup'].isin(['Toddler']), 0, df_covid['AgeGroup'])
df_covid['AgeGroup'] = np.where(df_covid['AgeGroup'].isin(['Children']), 1, df_covid['AgeGroup'])
df_covid['AgeGroup'] = np.where(df_covid['AgeGroup'].isin(['Pre-Teens']), 2, df_covid['AgeGroup'])
df_covid['AgeGroup'] = np.where(df_covid['AgeGroup'].isin(['Teenager']), 3, df_covid['AgeGroup'])
df_covid['AgeGroup'] = np.where(df_covid['AgeGroup'].isin(['Adult']), 4, df_covid['AgeGroup'])
df_covid['AgeGroup'] = np.where(df_covid['AgeGroup'].isin(['Senior']), 5, df_covid['AgeGroup'])
df_covid['AgeGroup'].unique()

#HealthStatus
#0 - Mild
#1 - Severe
#2 - Critical
#3 - Died
#4 - Asymptomatic
#5 - Recovered

df_covid['HealthStatus'] = np.where(df_covid['HealthStatus'].isin(['Mild']), 0, df_covid['HealthStatus'])
df_covid['HealthStatus'] = np.where(df_covid['HealthStatus'].isin(['Severe']), 1, df_covid['HealthStatus'])
df_covid['HealthStatus'] = np.where(df_covid['HealthStatus'].isin(['Critical']), 2, df_covid['HealthStatus'])
df_covid['HealthStatus'] = np.where(df_covid['HealthStatus'].isin(['Died']), 3, df_covid['HealthStatus'])
df_covid['HealthStatus'] = np.where(df_covid['HealthStatus'].isin(['Asymptomatic']), 4, df_covid['HealthStatus'])
df_covid['HealthStatus'] = np.where(df_covid['HealthStatus'].isin(['Recovered']), 5, df_covid['HealthStatus'])
df_covid['HealthStatus'].unique()

#RegionRes
#0 - NCR
#1 - Region I: Ilocos Region
#2 - Region II: Cagayan Valley
#3 - Region III: Central Luzon
#4 - Region IV-A: CALABARZON
#5 - Region IV-B: MIMAROPA
#6 - Region V: Bicol Region
#7 - Region VI: Western Visayas
#8 - Region VII: Central Visayas
#9 - Region VIII: Eastern Visayas
#10 - Region IX: Zamboanga Peninsula
#11 - Region X: Northern Mindanao
#12 - Region XI: Davao Region
#13 - Region XII: SOCCSKSARGEN
#14 - CAR
#15 - BARMM
#16 -CARAGA

df_covid['RegionRes'] = np.where(df_covid['RegionRes'].isin(['NCR']), 0, df_covid['RegionRes'])
df_covid['RegionRes'] = np.where(df_covid['RegionRes'].isin(['Region I: Ilocos Region']), 1, df_covid['RegionRes'])
df_covid['RegionRes'] = np.where(df_covid['RegionRes'].isin(['Region II: Cagayan Valley']), 2, df_covid['RegionRes'])
df_covid['RegionRes'] = np.where(df_covid['RegionRes'].isin(['Region III: Central Luzon']), 3, df_covid['RegionRes'])
df_covid['RegionRes'] = np.where(df_covid['RegionRes'].isin(['Region IV-A: CALABARZON']), 4, df_covid['RegionRes'])
df_covid['RegionRes'] = np.where(df_covid['RegionRes'].isin(['Region IV-B: MIMAROPA']), 5, df_covid['RegionRes'])
df_covid['RegionRes'] = np.where(df_covid['RegionRes'].isin(['Region V: Bicol Region']), 6, df_covid['RegionRes'])
df_covid['RegionRes'] = np.where(df_covid['RegionRes'].isin(['Region VI: Western Visayas']), 7, df_covid['RegionRes'])
df_covid['RegionRes'] = np.where(df_covid['RegionRes'].isin(['Region VII: Central Visayas']), 8, df_covid['RegionRes'])
df_covid['RegionRes'] = np.where(df_covid['RegionRes'].isin(['Region VIII: Eastern Visayas']), 9, df_covid['RegionRes'])
df_covid['RegionRes'] = np.where(df_covid['RegionRes'].isin(['Region IX: Zamboanga Peninsula']), 10, df_covid['RegionRes'])
df_covid['RegionRes'] = np.where(df_covid['RegionRes'].isin(['Region X: Northern Mindanao']), 11, df_covid['RegionRes'])
df_covid['RegionRes'] = np.where(df_covid['RegionRes'].isin(['Region XI: Davao Region']), 12, df_covid['RegionRes'])
df_covid['RegionRes'] = np.where(df_covid['RegionRes'].isin(['Region XII: SOCCSKSARGEN']), 13, df_covid['RegionRes'])
df_covid['RegionRes'] = np.where(df_covid['RegionRes'].isin(['CAR']), 14, df_covid['RegionRes'])
df_covid['RegionRes'] = np.where(df_covid['RegionRes'].isin(['BARMM']), 15, df_covid['RegionRes'])
df_covid['RegionRes'] = np.where(df_covid['RegionRes'].isin(['CARAGA']), 16, df_covid['RegionRes'])
df_covid['RegionRes'].unique()



# envoke the df_alumni.head command to review the result. This time you can see that all variables in the list are numbers.
df_covid.head()

# view the size of your dataset
df_covid.shape

# View the total size of your dataset
df_covid.size

#C. Removing Null Values
# Identify attributes with null values
df_covid.isnull().sum()

#display only the number of rows of data before deletion of null values
before_rows = df_covid.shape[0]
print(before_rows)

#Dropping the null values
df_covid.dropna(inplace =True)

after_rows = df_covid.shape[0]
print(after_rows)

# Reviewing the dataset to doouble check if there still null values
df_covid.isnull().sum()

#D Analyzing Dataset using Histogram
#examines the values of the orignal dataset and its histograms 
print(df_covid.head(5))
pd.DataFrame.hist(df_covid, figsize = [5,5])


#E. Rescale, Binarize and Normalization of the Given Dataset
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

scaler = MinMaxScaler(feature_range=(0,1))
rescaledDS = scaler.fit_transform(df_covid)
rescaledDF = pd.DataFrame(rescaledDS)

rescaledDF.head()

