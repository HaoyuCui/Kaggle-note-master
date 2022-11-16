# Titanic - Kaggle实战

> https://www.kaggle.com/c/titanic

### 引入数据

```Python
import pandas as pd
import seaborn as sns
import matplotlib as plt

data = pd.read_csv(path, index_col = 0) # 避免产生unnamed列
combine = [train_data, test_data]  # combine是方便对训练和验证数据统一处理
```

### 探索数据

```python
data.head()  # 打印前5行
data.info()  # 打印dtypes信息和数据数量
data.describe()  # 打印数据的统计分布
data.columes.values  # 打印行索引
data.isnull.sum()  # 为空数值的数量，将会返回列表
data.groupby('Pclass').size()
data.groupby('Pclass').Pclass.count()  # 两行含义等同
```

#### 选择多项数据：行 列

```Python
indices = [0, 1, 2, 3]
cols = ['Sex', 'Pclass', 'Survived']
df = data.loc[indices, cols]
```

### 数据可视化

```python
# 计数可视化：统计每一种数据的个数，replace替换图表中的显示情况
data['Survived'].replace({0:"died",1:"survived"}).value_counts().plot.pie()
sns.countplot(x = train_data["Survived"].replace({0:"died",1:"survived"})) 
# seaborn 的 countplot函数可以直接接收series类型数据
```

#### 如果要选择多项进行比较：如不同性别/舱位的情况，需要使用pandas细分

`df[['特征', '需要统计的信息']].groupby(['特征'])；.mean()` 计算概率；结果会以table的形式返回notebook

```python
str featue = "Pclass"
train_data[[feature, "Survived"]].groupby([feature], as_index=False).mean().sort_values(by='Survived',    			 ascending=False)

sns.barplot(data = train_data , x = feature , y = "Survived").set_title(f"{feature} Vs Survived")
```

seaborn 的 barplot函数可以直接接收series类型数据，但需要以string的格式告知需要作图的索引

#### 相关性矩阵

```python
sns.set(rc = {'figure.figsize':(10,6)})
sns.heatmap(train_data.corr(), annot = True, fmt='.2g',cmap= 'YlGnBu')
```

探索相关性，通过该方法打印出表格中各元素的相互关联性，正负区分正负相关

`sns.heatmap(dataframe.corr(), annot是否标记数字, 颜色)`

#### 分段统计

`sns.hisplot(dataframe, x=横轴名称, hue=需要堆叠对比的数据, binwidth=需要分段数据的step, ax=ax[i]分配到第几个子图)`

使用loc从数据中搜索 `dataframe.loc[dataframe[]=='']`

```python
plot , ax = plt.subplots(1 , 3 , figsize=(14,4)) # 子图1行3列

sns.histplot(data=data.loc[train_data["Pclass"]==1], x="Age", hue="Survived", binwidth=5,ax = ax[0], multiple = "stack").set_title("1-Pclass")
```

可以加入 `palette = sns.color_palette(["yellow" , "green"])` 修改颜色

### 数据清洗

删掉无需要的数据

```python
data.drop(columns = ["需要删除的列"] , inplace = True) 
data.drop(["需要删除的列"] , axis=1, inplace = True) 
```

处理含有异常的数据，这里先清洗再用清洗后数据的最大值填充含有NaN值的value，inplace=True替换原值

```python
data["value"].fillna(data["value"].dropna().max(), inplace=True)
```

数字化表格中的标签：匹配并进行类型转化 eg；如果重命名索引使用`rename(columns = dict or index = dict`
`)`

```python
dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int) 
```

数据的分块处理

直接分块，索引名为范围（不建议直接使用）

```python
train_data['AgeBand'] = pd.cut(train_data['Age'], 5)
```

对分块后的数据进行命名，`loc[]` 查找后直接命名，注意，不是 ~~loc()~~

```Python
dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
```

也可以根据数值使用`map("替换字典")`确定新列索引，得到独热数据

```Python
dataset['LargeF'] = dataset['FamilySize'].map(lambda s: 1 if s >= 5 else 0)
```

