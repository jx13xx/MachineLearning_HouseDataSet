import os
import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

Train_Data = pd.read_csv('C:/Users/jeenx/Desktop/Assignment2/Assignment_2_pack/Assignment_2_pack/train.csv')
Train_Data1 = pd.read_csv('C:/Users/jeenx/Desktop/Assignment2/Assignment_2_pack/Assignment_2_pack/train.csv')
print(Train_Data.head(10))

print(Train_Data.info())
print(Train_Data.describe())

print('The total number of Missing/Null data is',Train_Data.isna().sum().sum())
print('Shape of training data is:', Train_Data.shape)
Nulls = pd.isnull(Train_Data).sum()
print(Nulls[Nulls>0])

print(Train_Data.columns)
Y = Train_Data.SalePrice
Train_Data.drop(['Id','MiscFeature', 'PoolQC','Fence','Alley','SalePrice'], axis = 1, inplace = True)

X = Train_Data
print(X.head(10))

Nulls1 = pd.isnull(X).sum()
print(Nulls1[Nulls1>0])

print(X['BsmtQual'].head(10))
print(X['BsmtQual'].describe())

#Can be Median or Mean 
from sklearn.impute import SimpleImputer 
imr = SimpleImputer(missing_values=np.nan, strategy='median')
imr = imr.fit(X[['LotFrontage']])
X['LotFrontage'] = imr.transform(X[['LotFrontage']])

imr1 = SimpleImputer(missing_values=np.nan, strategy='median')
imr1 = imr1.fit(X[['MasVnrArea']])
X['MasVnrArea'] = imr1.transform(X[['MasVnrArea']])

imr2 = SimpleImputer(missing_values=np.nan, strategy='median')
imr2 = imr2.fit(X[['GarageYrBlt']])
X['GarageYrBlt'] = imr1.transform(X[['GarageYrBlt']])

# Filling the NA Values with NA and Appropriate Replacements
X['MasVnrType'].fillna("None", inplace = True)
X['BsmtQual'].fillna("NA", inplace = True)
X['BsmtCond'].fillna("NA", inplace = True)
X['BsmtExposure'].fillna("NA", inplace = True)
X['BsmtFinType1'].fillna("NA", inplace = True)
X['BsmtFinType2'].fillna("NA", inplace = True)
X['Electrical'].fillna("SBrkr", inplace = True)
X['FireplaceQu'].fillna("TA", inplace = True)
X['GarageType'].fillna("NA", inplace = True)
X['GarageFinish'].fillna("NA", inplace = True)
X['GarageQual'].fillna("NA", inplace = True)
X['GarageCond'].fillna("NA", inplace = True)

print(X.columns)
Nulls2 = pd.isnull(X).sum()
print(Nulls2[Nulls2>0])

#Label Encoding 
labelencoder=LabelEncoder()

X['MSZoning'] = labelencoder.fit_transform(X['MSZoning'].astype(str))
X['Street'] = labelencoder.fit_transform(X['Street']) 
X['LotShape'] = labelencoder.fit_transform(X['LotShape']) 
X['LandContour'] = labelencoder.fit_transform(X['LandContour']) 
X['Utilities'] = labelencoder.fit_transform(X['Utilities']) 
X['LotConfig'] = labelencoder.fit_transform(X['LotConfig'])  
X['LandSlope'] = labelencoder.fit_transform(X['LandSlope'])    
X['Neighborhood'] = labelencoder.fit_transform(X['Neighborhood']) 
X['Condition1'] = labelencoder.fit_transform(X['Condition1'])   
X['Condition2'] = labelencoder.fit_transform(X['Condition2'])   
X['BldgType'] = labelencoder.fit_transform(X['BldgType'])   
X['HouseStyle'] = labelencoder.fit_transform(X['HouseStyle'])   
X['RoofStyle'] = labelencoder.fit_transform(X['RoofStyle'])   
X['RoofMatl'] = labelencoder.fit_transform(X['RoofMatl'])   
X['Exterior1st'] = labelencoder.fit_transform(X['Exterior1st'].astype(str))  
X['Exterior2nd'] = labelencoder.fit_transform(X['Exterior2nd'].astype(str))  
X['MasVnrType'] = labelencoder.fit_transform(X['MasVnrType'])   
X['ExterQual'] = labelencoder.fit_transform(X['ExterQual'])  
X['ExterCond'] = labelencoder.fit_transform(X['ExterCond'])   
X['Foundation'] = labelencoder.fit_transform(X['Foundation'])   
X['BsmtQual'] = labelencoder.fit_transform(X['BsmtQual'])   
X['BsmtCond'] = labelencoder.fit_transform(X['BsmtCond'])   
X['BsmtExposure'] = labelencoder.fit_transform(X['BsmtExposure'])   
X['BsmtFinType1'] = labelencoder.fit_transform(X['BsmtFinType1'])   
X['BsmtFinType2'] = labelencoder.fit_transform(X['BsmtFinType2'])
X['BsmtFinSF1'] = labelencoder.fit_transform(X['BsmtFinType1'])   
X['BsmtFinSF2'] = labelencoder.fit_transform(X['BsmtFinType2'])     
X['Heating'] = labelencoder.fit_transform(X['Heating'])   
X['HeatingQC'] = labelencoder.fit_transform(X['HeatingQC'])   
X['CentralAir'] = labelencoder.fit_transform(X['CentralAir'])   
X['Electrical'] = labelencoder.fit_transform(X['Electrical'])   
X['KitchenQual'] = labelencoder.fit_transform(X['KitchenQual'].astype(str))
X['Functional'] = labelencoder.fit_transform(X['Functional'].astype(str))
X['FireplaceQu'] = labelencoder.fit_transform(X['FireplaceQu'])  
X['GarageType'] = labelencoder.fit_transform(X['GarageType'])  
X['GarageFinish'] = labelencoder.fit_transform(X['GarageFinish'])   
X['GarageQual'] = labelencoder.fit_transform(X['GarageQual'])  
X['GarageCond'] = labelencoder.fit_transform(X['GarageCond'])   
X['PavedDrive'] = labelencoder.fit_transform(X['PavedDrive']) 
X['SaleType'] = labelencoder.fit_transform(X['SaleType'].astype(str))   
X['SaleCondition'] = labelencoder.fit_transform(X['SaleCondition'])  

print(X.head(50))

#Scaling using Standard Scaler 
scaler = StandardScaler()
scaler.fit(X)
#change y to natural log 
Y = np.log(Y)

#Train Test Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.1, random_state=42)
# Building Function for Print
from sklearn import metrics
def print_evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('--------------------------------')
def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square

#Linear Regression
from sklearn import linear_model
lr = LinearRegression(normalize= True)
model = lr.fit(X_train, y_train)

test_pred = lr.predict(X_test)
train_pred = lr.predict(X_train)
print('Test set evaluation:\n-----------------------')
print_evaluate(y_test, test_pred)
print('Train set evaluation:\n------------------------')
print_evaluate(y_train, train_pred)

results_df = pd.DataFrame(data=[["Linear Regression", *evaluate(y_test, test_pred)]], 
                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square'])
print(results_df)

#Ridge Regression
from sklearn.linear_model import Ridge
model = Ridge(alpha=100, solver='cholesky', tol=0.0001, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
test_pred = model.predict(X_test)
train_pred = model.predict(X_train)
print('Test set evaluation:\n------------------------')
print_evaluate(y_test, test_pred)
print('------------------------')
print('Train set evaluation:\n-----------------------')
print_evaluate(y_train, train_pred)

results_df_2 = pd.DataFrame(data=[["Ridge Regression", *evaluate(y_test, test_pred)]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square'])
results_df = results_df.append(results_df_2, ignore_index=True)
print(results_df)

#ElasticNet
from sklearn.linear_model import ElasticNet
model = ElasticNet(alpha=0.1, l1_ratio=0.9, selection='random', random_state=42)
model.fit(X_train, y_train)
test_pred = model.predict(X_test)
train_pred = model.predict(X_train)
print('Test set evaluation:\n ------------------------')
print_evaluate(y_test, test_pred)
print('--------------------------------')
print('Train set evaluation:\n-----------------------')
print_evaluate(y_train, train_pred)

results_df_2 = pd.DataFrame(data=[["Elastic Net Regression", *evaluate(y_test, test_pred)]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square'])
results_df = results_df.append(results_df_2, ignore_index=True)
print(results_df)

#GridSearch for Ridge Regression
from sklearn.model_selection import GridSearchCV 
param_grid = {'alpha': [0.1,1, 10, 100], 'solver': ['svd', 'cholesky','auto','lsqr','sparse_cg']}
RidgeGS = GridSearchCV(Ridge(), param_grid, refit = True, verbose = 2) 
RidgeGS.fit(X_train, y_train) 
pred = RidgeGS.predict(X_test)
test_pred = RidgeGS.predict(X_test)
train_pred = RidgeGS.predict(X_train)
print('Test set evaluation:\n------------------------')
print_evaluate(y_test, test_pred)
print('------------------------')
print('Train set evaluation:\n-----------------------')
print_evaluate(y_train, train_pred)

results_df_2 = pd.DataFrame(data=[["Ridge Regression with GridSearch", *evaluate(y_test, test_pred)]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square'])
results_df = results_df.append(results_df_2, ignore_index=True)
print(results_df)

#GridSearch for ElasticNet
param_grid = {'alpha': [0.1,1, 10, 100], 'l1_ratio': [0.1,0.3,0.5,0.7,0.9], 'selection': ['cyclic', 'random']}
ElasticNGS = GridSearchCV(ElasticNet(), param_grid, refit = True, verbose = 2) 
ElasticNGS.fit(X_train, y_train)
pred = ElasticNGS.predict(X_test)

test_pred = ElasticNGS.predict(X_test)
train_pred = ElasticNGS.predict(X_train)

print('Test set evaluation:\n------------------------')
print_evaluate(y_test, test_pred)
print('------------------------')
print('Train set evaluation:\n-----------------------')
print_evaluate(y_train, train_pred) 

results_df_2 = pd.DataFrame(data=[["ElasticNet with GridSearch", *evaluate(y_test, test_pred)]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square'])
results_df = results_df.append(results_df_2, ignore_index=True)
print(results_df)

#PCR - Principal Component Regression
X_PCA = Train_Data1
X_PCA.drop(['Id','MiscFeature', 'PoolQC','Fence','Alley','FireplaceQu'], axis = 1, inplace = True)

from sklearn.impute import SimpleImputer 
imr = SimpleImputer(missing_values=np.nan, strategy='median')
imr = imr.fit(X_PCA[['LotFrontage']])
X_PCA['LotFrontage'] = imr.transform(X_PCA[['LotFrontage']])

imr1 = SimpleImputer(missing_values=np.nan, strategy='median')
imr1 = imr1.fit(X_PCA[['MasVnrArea']])
X_PCA['MasVnrArea'] = imr1.transform(X_PCA[['MasVnrArea']])

imr2 = SimpleImputer(missing_values=np.nan, strategy='median')
imr2 = imr2.fit(X_PCA[['GarageYrBlt']])
X_PCA['GarageYrBlt'] = imr1.transform(X_PCA[['GarageYrBlt']])

# Filling the NA Values with NA and Appropriate Replacements
#X['MasVnrType'].fillna("None", inplace = True)
#X['BsmtQual'].fillna("TA", inplace = True)
#X['BsmtCond'].fillna("TA", inplace = True)
#X['BsmtExposure'].fillna("No", inplace = True)
#X['BsmtFinType1'].fillna("UnF", inplace = True)
#X['BsmtFinType2'].fillna("UnF", inplace = True)
#X['Electrical'].fillna("SBrkr", inplace = True)
#X['GarageType'].fillna("Attchd", inplace = True)
#X['GarageFinish'].fillna("Unf", inplace = True)
#X['GarageQual'].fillna("TA", inplace = True)
#X['GarageCond'].fillna("TA", inplace = True)
X_PCA['MasVnrType'].fillna("None", inplace = True)
X_PCA['BsmtQual'].fillna("NA", inplace = True)
X_PCA['BsmtCond'].fillna("NA", inplace = True)
X_PCA['BsmtExposure'].fillna("NA", inplace = True)
X_PCA['BsmtFinType1'].fillna("NA", inplace = True)
X_PCA['BsmtFinType2'].fillna("NA", inplace = True)
X_PCA['Electrical'].fillna("SBrkr", inplace = True)
X_PCA['GarageType'].fillna("NA", inplace = True)
X_PCA['GarageFinish'].fillna("NA", inplace = True)
X_PCA['GarageQual'].fillna("NA", inplace = True)
X_PCA['GarageCond'].fillna("NA", inplace = True)

num_train = X_PCA._get_numeric_data()
num_train.columns

num_corr=num_train.corr()
plt.subplots(figsize=(13,10))
sns.heatmap(num_corr,vmax =.8 ,square = True)


k = 14
cols = num_corr.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(num_train[cols].values.T)
sns.set(font_scale=1.35)
f, ax = plt.subplots(figsize=(10,10))
hm=sns.heatmap(cm, annot = True,vmax =.8, yticklabels=cols.values, xticklabels = cols.values)


from sklearn.preprocessing import StandardScaler
X_PCA = pd.get_dummies(X_PCA)
Y_PCA = X_PCA.SalePrice
X_PCA.drop(['SalePrice'], axis = 1, inplace = True)
scaler = StandardScaler()
scaler.fit(X_PCA)                
X_PCA = scaler.transform(X_PCA)

from sklearn.decomposition import PCA
pca_hp = PCA(30)
X_PCA = pca_hp.fit_transform(X_PCA)
np.exp(pca_hp.explained_variance_ratio_)

X_train_PCA, X_test_PCA, y_train_PCA, y_test_PCA = train_test_split(X_PCA,Y_PCA, test_size=0.1, random_state=42)

lr1 = LinearRegression(normalize= True)
model1 = lr1.fit(X_train_PCA, y_train_PCA)

test_pred_pca = lr1.predict(X_test_PCA)
train_pred_pca = lr1.predict(X_train_PCA)
print('Test set evaluation:\n-----------------------')
print_evaluate(y_test_PCA, test_pred_pca)
print('Train set evaluation:\n------------------------')
print_evaluate(y_train_PCA, train_pred_pca)
results1_df = pd.DataFrame(data=[["Principal Component Regression", *evaluate(y_test_PCA, test_pred_pca)]], 
                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square'])
results_df = results_df.append(results1_df, ignore_index=True)
print(results_df)

#Swiss Roll
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_swiss_roll

X_PCA, Y_PCA = make_swiss_roll(n_samples=1460, noise=0.2, random_state=42)

axes = [-11.5, 14, -2, 23, -12, 15]

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_PCA[:, 0], X_PCA[:, 1], X_PCA[:, 2], c=Y_PCA, cmap=plt.cm.hot)
ax.view_init(10, -70)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])

plt.show()