# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![436853431-31733353-2e44-45bb-9633-c6c82b16d3f5](https://github.com/user-attachments/assets/076120af-c445-4cc8-9785-1244e570c8c0)

```
data.isnull().sum()
```
![436853560-9833529a-86bd-4abc-a554-d1f3a9de7fd4](https://github.com/user-attachments/assets/eba26a9e-af61-4f04-92f8-3a2943d61f1e)

```
missing=data[data.isnull().any(axis=1)]
missing
```
![436853894-922e37fe-ee22-456d-ad41-5bb83417cc8e](https://github.com/user-attachments/assets/50b73888-5cd3-4b4e-ba13-5c07cf4cdeb7)

```
data2=data.dropna(axis=0)
data2
```
![436853959-acb72b11-5e15-4c71-ab35-6e73ef19f395](https://github.com/user-attachments/assets/5752d95c-5ee4-43ba-9609-2f077a3eb10f)

```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![436854144-511963d8-0bdb-456b-ade7-0046592d803e](https://github.com/user-attachments/assets/0bb2da6e-ba8c-413f-b3d4-9c92795df540)

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![436854334-a5a33e5f-99e2-42a1-8fc2-704bf215adee](https://github.com/user-attachments/assets/490d6a1d-68ed-4605-8fad-e77c891adfa9)

```
data2
```
![436854623-31047988-6207-4e4c-923e-f3133e8346e0](https://github.com/user-attachments/assets/2df2c662-a4cd-4c7e-a520-f967fc0c0309)

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![436854706-ebbf7c45-d235-406b-8cb8-7b79a999f631](https://github.com/user-attachments/assets/2ffa8851-3018-4f99-bf55-a57e4079f282)

```
columns_list=list(new_data.columns)
print(columns_list)
```
![436855272-3deeee10-5767-4136-b055-29ad3d6368df](https://github.com/user-attachments/assets/e9fd21ec-355d-458e-ac3a-d5716abe7daa)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![436855375-1b389e6e-d7c1-40d7-94ed-292a33621c82](https://github.com/user-attachments/assets/5f441028-4d94-41ba-a5b6-e73e8311cbf1)

```
y=new_data['SalStat'].values
print(y)
```
![436855537-bd94c421-187d-497e-b8fa-23125ff8563a](https://github.com/user-attachments/assets/59250349-0117-42a3-9cc8-b98be6521e34)

```
x=new_data[features].values
print(x)
```
![436855708-75b1dbf6-a90d-4e29-8870-cdb39581515c](https://github.com/user-attachments/assets/06c9ac7a-6926-4fb4-bd70-7a9545e810c0)

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![436855803-896e34f0-99d5-473c-8ad8-a6b6347b9743](https://github.com/user-attachments/assets/2543f919-786f-4799-a7b1-9f92b631ff11)

```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![436855898-5950532e-ad3c-4911-90bf-0a07c2b23a31](https://github.com/user-attachments/assets/44322935-d116-497c-a9f6-1aacaeb9b022)

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![436856006-46db0e4c-eba3-460f-b0e3-1639cddfe334](https://github.com/user-attachments/assets/fb4bfcd3-7f81-41bd-90cf-8ee1004704cd)

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![436856115-3ced9a76-9bc9-4780-8dda-bf5405413afe](https://github.com/user-attachments/assets/fc077826-0c7e-497d-ae82-039aba1457ea)

```
data.shape
```
![436856221-9922476a-30f7-45c9-bb9f-db751d246214](https://github.com/user-attachments/assets/1fe044eb-fd93-473f-bfbf-3bb9658a1c5d)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![436856364-df7adfb8-bc99-4cf1-9c18-2df9a87cd73d](https://github.com/user-attachments/assets/35254ec6-12e6-489f-b911-ad18cd74c352)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![436856470-167213bb-f604-48e5-9abe-f08cbaf45822](https://github.com/user-attachments/assets/adb8b4fa-5f85-485e-b4c0-e9d6bb9c8870)

```
tips.time.unique()
```
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

![436856648-049e91cd-0e5e-408c-9497-74a900918e5d](https://github.com/user-attachments/assets/8dbd3f42-d282-4e30-8a6e-d98f25471203)

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![436856781-cc318136-926b-451d-9e63-14b63a7e3911](https://github.com/user-attachments/assets/18ef6303-8d5d-43e2-808f-b44a2cc05148)

# RESULT:
    Thus, Feature selection and Feature scaling has been used on thegiven dataset.
