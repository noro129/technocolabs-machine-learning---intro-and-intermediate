#Intermediate Machine Learning
##Missing Values

#1
num_rows = 1168
num_cols_with_missing = 3
tot_missing = 212+6+58

#2
col_with_missed_data=[col for col in X_train if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(col_with_missed_data,axis=1)
reduced_X_valid = X_valid.drop(col_with_missed_data,axis=1)

#3
imputer=SimpleImputer()
imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(imputer.transform(X_valid))

imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

#4
final_X_train = X_train.fillna(0)
final_X_valid = X_valid.fillna(0)

final_X_test = X_test.fillna(0)

preds_test = model.predict(final_X_test)

##Categorical Variables

#1
categorical_cols=[col for col in X_train if X_train[col].dtype=='object']
drop_X_train = X_train.drop(categorical_cols,axis=1)
drop_X_valid = X_valid.drop(categorical_cols,axis=1)

#2
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)


ordinal_encoder = OrdinalEncoder()
label_X_train[good_label_cols]=ordinal_encoder.fit_transform(X_train[good_label_cols])
label_X_valid[good_label_cols]=ordinal_encoder.transform(X_valid[good_label_cols])

#3
high_cardinality_numcols = 3
num_cols_neighborhood = 25


OH_entries_added = 990000
label_entries_added = 0

#4
OH_encoder=OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train=pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_vali= pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

##Pipelines

#1
numerical_transformer =SimpleImputer(strategy='most_frequent')

categorical_transformer =Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


model = RandomForestRegressor(n_estimators=100, random_state=0)

#2
preds_test = my_pipeline.predict(X_test)

##Cross-Validation

#1
mypipeline=Pipeline(steps=[('imputer',SimpleImputer()),
('model',RandomForestRegressor(n_estimators=n_estimators,random_state=0))])
score=-1*cross_val_score(mypipeline,X,y,cv=3,scoring='neg_mean_absolute_error')
return score.mean()

#2
results = {i*50 : get_score(i*50) for i in range(1,9)}

#3
n_estimators_best = 200

##XGBoost

#1
my_model_1 = XGBRegressor(random_state=0)

my_model_1.fit(X_train,y_train)

predictions_1 = my_model_1.predict(X_valid)
mae_1 = mean_absolute_error(y_valid,predictions_1)

#2
my_model_2 = XGBRegressor(random_state=0,n_estimators=500,learning_rate=0.05)

my_model_2.fit(X_train,y_train)

predictions_2 = my_model_2.predict(X_valid)

mae_2 = mean_absolute_error(y_valid,predictions_2)

#3
my_model_3 = XGBRegressor(random_state=0,n_estimators=20,learning_rate=0.4)

my_model_3.fit(X_train,y_train)

predictions_3 = my_model_3.predict(X_valid)

mae_3 = mean_absolute_error(y_valid,predictions_3)