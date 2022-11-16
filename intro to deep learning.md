# Intro to Deep Learning

> https://www.kaggle.com/code/ryanholbrook/a-single-neuron

```Python
from tensorflow import keras
from tensorflow.keras import layers
```

### 隐藏层

```Python
model = keras.Sequential([
    layers.Dense(units=4, activation='relu', input_shape=[2]),
  	layers.Dropout(rate=0.3),
    layers.Dense(units=3, activation='relu'),
  	layers.Dropout(rate=0.3),
    layers.Dense(units=1),
])
```

使用`w, b = model.weights`打印权重

可以使用`layers.Activation('relu')`单独将激活层置于一层

### 优化器和损失函数

```Python
model.compile(
    optimizer='adam',
    loss='mae'
)
```

### 模型训练

```Python
history = model.fit(
    X, y, 
  	validation_data=(X_valid, y_valid),
    batch_size=128,
    epochs=200,
  	callbacks=[early_stopping],  # in earlystopping
  	verbose=0,   # don't print much 
)
```

作图

```Python
history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['loss', 'val_loss']].plot()  # 从第一个epoch开始作图
```

#### earlystopping

```Python
from tensorflow.keras import callbacks
```

`patience`等待5个epoch之后采取

`min_delta`最小允许变化

`Restore_best_weights`保持最佳权重

```Python
early_stopping = callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True
)
```

#### DropOut

使用`layers.Dropout(rate)`

#### BatchNormalization

使用`layers.BatchNormalization(input_shape=input_shape)`

### Binary Classification

```Python
model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['binary_accuracy'],  # 数组，可以加入多个评价指标
)
```

作图

```Python
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot();
```

