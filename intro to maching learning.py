#Intro to Machine Learning
##Basic Data Exploration

#1
home_data = pd.read_csv(iowa_file_path)

#2
home_data.describe()

avg_lot_size = 10517
newest_home_age = 12

##Your First Machine Learning Model

#1
home_data.columns

y = home_data.SalePrice

#2
feature_names=['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X = home_data[feature_names]

#3
X.describe()

X.head()

from sklearn.tree import DecisionTreeRegressor

iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(X,y)

#4
predictions = iowa_model.predict(X)
print(predictions)

y.head()

##Model Validation

#1
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=1)

#2
from sklearn.tree import DecisionTreeRegressor
iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(train_X,train_y)

#3
val_predictions = iowa_model.predict(val_X)

print(val_predictions[:5])
print(val_y[:5])

#4
from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_y,val_predictions)

print(val_mae)

##Underfitting and Overfitting

#1
mae=get_mae(5, train_X, val_X, train_y, val_y)
tree_size=5
for i in candidate_max_leaf_nodes:
    if get_mae(i, train_X, val_X, train_y, val_y)<mae:
        mae=get_mae(i, train_X, val_X, train_y, val_y)
        tree_size=i

best_tree_size = tree_size

#2
final_model=DecisionTreeRegressor(max_leaf_nodes=best_tree_size,random_state=1)

final_model.fit(X, y)

##Random Forests

#1
rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X,train_y)
rf_val_mae = mean_absolute_error(val_y,rf_model.predict(val_X))

