#import modules
from os import X_OK
import pandas as pd
import numpy as np
from pyparsing import col
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


#import data
path_to_dataset = "./spaceship-titanic-data/train.csv"
X=pd.read_csv(path_to_dataset)
whole_train_y=X['Transported']

#delete unnecessary columns
categorical_data = X.drop(columns=['Transported','Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Cabin', 'Name'])

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

X = combined_X.drop(columns=['Name','PassengerId', 'Cabin', 'Transported'])

imp = SimpleImputer(missing_values=np.NaN, strategy='mean')
whole_train_X = pd.DataFrame(imp.fit_transform(X))
whole_train_X.columns = X.columns
whole_train_X.index = X.index

train_X, test_X, train_y, test_y = train_test_split(
    whole_train_X, whole_train_y, random_state=1, test_size=0.2)

