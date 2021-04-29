import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sklearn
from sklearn.impute import SimpleImputer
from sklearn import metrics

import settings
from settings import *


class regression:
    temp = None
    LOGS_SEPARATOR = settings.logged

    def __init__(self):
        self.run_main()
        self.X
        self.Y

    def run_main(self):
        try:
            Train_Data = pd.read_csv('C:/Users/jeenx/Desktop/Assignment2/Assignment_2_pack/Assignment_2_pack/train.csv')
            n = Train_Data
            self.temp = self.read(n) if (Train_Data is not None) else print("")
        except:
            print('File not Found exception.')

    def read(self, n):
        self.display_data(n)
        self.display_info(n)
        # self.display_describe(n)
        # get the total numbers of cells with NA values
        print(self.LOGS_SEPARATOR)
        print('Total Number of NULL Data Before DATA CLEAN:', n.isna().sum().sum())
        print('Shape of Training Data: ', n.shape)
        # FINDING_THE TOTALE NULL BEFORE DROPPING UNCESSARY FIELDS
        Nulls = pd.isnull(n).sum()
        Nulls[Nulls > 0]

        self.Y = n.SalePrice

        n.drop(['Id', 'MiscFeature', 'PoolQC', 'Fence', 'Alley', 'SalePrice'], axis=1, inplace=True)
        self.X = n
        # self.display_data(X)
        print('Total Number of NULL Data After DATA CLEAN:', n.isna().sum().sum())
        print('Shape of Training Data: ', n.shape)
        print(self.LOGS_SEPARATOR)

        # FINDING THE TOTAL NUM AFTER DROPPING UNCESSSARY COLUMNS
        Nulls1 = pd.isnull(self.X).sum()
        Nulls1[Nulls1 > 0]
        print(Nulls1)

        print(self.X['BsmtQual'].head(10))
        print(self.X['BsmtQual'].describe())

        # Can be Median or Mean
        self.transform_data(self.X)

        # Encode target labels with value between 0 and n_classes-1.
        self.temp = print('ENCODING_DONE') if(self.data_encode(self.X) is True) else print('ENCODING_FAILED')
        self.display_data(self.X)

        scaler = StandardScaler()
        scaler.fit(self.X)
        # change y to natural log
        self.Y = np.log(self.Y)
        self.display_data(self.Y)
        print('Y data above')

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.1, random_state=42)
        # print('hello')
        print('Train Split')
        try:
            self.temp = "TEST TRAINING COMPLETED" if (self.print_evaluate(True, y_test) is True) else print("!!!!TEST TRAINED FAILED")
        except Exception as ex:
            raise RuntimeError("Something bad happened") from ex
            print('Training Exception')

    def transform_data(self, X):
        print('transform')
        imr = SimpleImputer(missing_values=np.nan, strategy='median')
        imr = imr.fit(X[['LotFrontage']])
        X['LotFrontage'] = imr.transform(X[['LotFrontage']])


        imr1 = SimpleImputer(missing_values=np.nan, strategy='median')
        imr1 = imr1.fit(X[['MasVnrArea']])
        X['MasVnrArea'] = imr1.transform(X[['MasVnrArea']])

        imr2 = SimpleImputer(missing_values=np.nan, strategy='median')
        imr2 = imr2.fit(self.X[['GarageYrBlt']])
        X['GarageYrBlt'] = imr1.transform(X[['GarageYrBlt']])


        self.temp = print('COMPLETED FILL NA') if (self.fill_Na(X) is True) else print("!!!!FILL NA FAILED")
        # self.display_data(X)
        # print(X['GarageType'].head(10))
        # print(X['GarageType'].describe())

        print('end-transform')

    def data_encode(self,X):
        print('Inside Data Encoding')
        labelencoder = LabelEncoder()
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
        return True


    def fill_Na(self,X):
        # Filling the NA Values with NA and Appropriate Replacements
        print('inside fille NA')
        X['MasVnrType'].fillna("None", inplace=True)
        X['BsmtQual'].fillna("NA", inplace=True)
        X['BsmtCond'].fillna("NA", inplace=True)
        X['BsmtExposure'].fillna("NA", inplace=True)
        X['BsmtFinType1'].fillna("NA", inplace=True)
        X['BsmtFinType2'].fillna("NA", inplace=True)
        X['Electrical'].fillna("SBrkr", inplace=True)
        X['FireplaceQu'].fillna("TA", inplace=True)
        X['GarageType'].fillna("NA", inplace=True)
        X['GarageFinish'].fillna("NA", inplace=True)
        X['GarageQual'].fillna("NA", inplace=True)
        X['GarageCond'].fillna("NA", inplace=True)
        print('done till fill NA')
        return True

    def display_data(self, data, default=10):
        # Listen/ Display  the current value of the data
        # Default value = 10
        print('inside display')
        return print(data.head(default))

    def display_info(self, data):
        # Display the info data value of the data
        print('inside info')
        return print(data.info())

    def display_describe(self, data):
        # Describe the data of
        print('inside describe')
        return print(data.describe())

    def print_evaluate(self, true=None, predicted=None):
        print('Evaluation')
        mae = metrics.mean_absolute_error(true, predicted)
        mse = metrics.mean_squared_error(true, predicted)
        rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
        r2_square = metrics.r2_score(true, predicted)
        print('MAE:', mae)
        print('MSE:', mse)
        print('RMSE:', rmse)
        print('R2 Square', r2_square)
        print('--------------------------------')
        return True;

p1 = regression();
# p1.run_main();
