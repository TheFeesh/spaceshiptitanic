from telnetlib import XDISPLOC
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from LogisticRegression import *

#import data
path_to_dataset = "./spaceship-titanic-data/test.csv"
X=pd.read_csv(path_to_dataset)
IDs = X['PassengerId']

categorical_data = X.drop(columns=['PassengerId','Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Cabin', 'Name'])

categorical_data['HomePlanet'] = categorical_data['HomePlanet'].astype('category')
categorical_data['CryoSleep'] = categorical_data['CryoSleep'].astype('category')
categorical_data['Destination'] = categorical_data['Destination'].astype('category')
categorical_data['VIP'] = categorical_data['VIP'].astype('category')

categorical_data['HomePlanet_new']=categorical_data['HomePlanet'].cat.codes
categorical_data['CryoSleep_new']=categorical_data['CryoSleep'].cat.codes
categorical_data['Destination_new']=categorical_data['Destination'].cat.codes
categorical_data['VIP_new']=categorical_data['VIP'].cat.codes

enc = OneHotEncoder()

enc_data = pd.DataFrame(enc.fit_transform(categorical_data[['HomePlanet_new', 'CryoSleep_new', 'Destination_new', 'VIP_new']]).toarray())

combined_X = X.drop(columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP'])
combined_X = combined_X.join(enc_data)

X = combined_X.drop(columns=['Name','PassengerId', 'Cabin'])

imp = SimpleImputer(missing_values=np.NaN, strategy='mean')
finalX = pd.DataFrame(imp.fit_transform(X))
finalX.columns = X.columns
finalX.index = X.index

# print(finalX)



