# home_data数据集 - 一个简单的demo

> https://www.kaggle.com/learn/intro-to-machine-learning

### 引入机器学习库

```Python
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
```

### 训练集和验证集设置

```Python
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]
```

划分数据

```Python
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
```

### 拟合数据

指定模型

```Python
iowa_model = DecisionTreeRegressor(random_state=1)
```

拟合数据

```Python
iowa_model.fit(train_X, train_y)
```

### 验证模型

```python
val_predictions = iowa_model.predict(val_X)
```

使用MAE验证

```Python
from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_y, val_predictions)
```

使用不同深度的决策树验证

```Python
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)
```

得到最后模型

```Python
final_model = DecisionTreeRegressor(max_leaf_nodes=100 ,random_state=1)
```

