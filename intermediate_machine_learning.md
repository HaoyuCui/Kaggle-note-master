# Machine Learning Intermediate

> https://www.kaggle.com/code/haoyuchui/exercise-introduction

### 处理缺失数据

#### dropna

删除包含NaN值的行

```Python
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
```

直接删除列：`inplace=True`代表本行可以直接使用，不返回值

```Python
X_full.drop(['SalePrice'], axis=1, inplace=True)
```

排除特定类型的数据

```Python
X = X_full.select_dtypes(exclude=['object'])
```

**获取包含空值的列**

```Python
cols_with_missing = [col for col in X_train.columns 
                    if X_train[col].isnull().any()]
```

#### inputation

```Python
from sklearn.impute import SimpleImputer
inputer = SimpleImputer()
```

需要对训练集先做fit，再进行`imputer.transform`

```Python
imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(imputer.transform(X_valid))
```

由于inputation会移除列名，需要重新命名

```Python
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns
```

### 处理类型数据

#### 序列编码

**获取包含类型的列**

```Python
object_cols = [col for col in X_train.columns 
               if X_train[col].dtype == "object"]
```

不是所有类型数据都适合编码，验证集中的类型数据一定需要是训练集的子集

```Python
good_label_cols = [col for col in object_cols if 
                   set(X_valid[col]).issubset(set(X_train[col]))]
```

非子集的集合，需要`drop()`

```Python
bad_label_cols = list(set(object_cols)-set(good_label_cols))
label_X = X.drop(bad_label_cols, axis=1)
```

序列编码过程，无需取代列，使用序列编码替换

```Python
from sklearn.preprocessing import OrdinalEncoder
label_X_train[good_label_cols] = ordinal_encoder.fit_transform(X_train[good_label_cols])
label_X_valid[good_label_cols] = ordinal_encoder.transform(X_valid[good_label_cols]) 
```

#### 独热编码

**获取类型数据类型小于10的列**

```Python
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]
```

独热编码过程

```Python
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(one_hot_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(one_hot_encoder.transform(X_valid[low_cardinality_cols]))
```

onehot会移除列名，需要使用`index`传回

```Python
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index
```

丢弃不需要的列

```Python
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)
```

onehot是新增列，将它与原数据组合

```Python
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
```

### Pipeline

选取所有类型数据的列：类型不超过10种适合编码

```Python
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and 
                    X_train_full[cname].dtype == "object"]
```

选取所有数字型数据的列

```Python
numerical_cols = [cname for cname in X_train_full.columns if 
                X_train_full[cname].dtype in ['int64', 'float64']]
```

适合统计的列

```Python
feature_cols = categorical_cols + numerical_cols
```

流程

```Python
# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)
```

构建模型和Pipeline

```Python
model = RandomForestRegressor()
# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)
```

#### CrossValidation

```Python
def get_score(n_estimators, my_pipeline):
    scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=3,
                              scoring='neg_mean_absolute_error')
    return scores.mean()
```

#### XGBoost

`n_estimator`参数表示建模周期的次数，过低将导致欠拟合，过高将导致过拟合，也可以设置一个较高的该值，然后使用`fit()`函数中的参数`early_stopping_rounds`提前停止迭代，其典型值为5

`n_jobs`在处理大型数据集中较为有效，其值设置为内核数

```Python
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)
```

