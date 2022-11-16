# Computer Vision

> https://www.kaggle.com/code/ryanholbrook/the-convolutional-classifier

### 载入预训练模型

```Python
import tensorflow_hub as hub

pretrained_base = tf.keras.models.load_model(
    '../input/cv-course-models/cv-course-models/inceptionv1'
)
pretrained_base.trainable = False  # 预训练模型将不会被重新训练
```

#### Attach Head

```Python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    pretrained_base,
    layers.Flatten(),
    layers.Dense(units=6, activation='relu'),
    layers.Dense(units=1, activation='sigmoid')
])

```

#### 训练

```Python
history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=30,
)
```

### 卷积

#### 自定义Kernel并在image上进行卷积操作

```Python
kernel = tf.constant([
    [1,2,1],
    [2,7,2],
    [1,2,1]    
])
image_filter = tf.nn.conv2d(
    input=image,
    filters=kernel,
    strides=1,
    padding='VALID',  # 'SAME'
)
image_detect = tf.nn.relu(image_filter)
```

#### 自定义池化

```Python
image_condense = tf.nn.pool(
    input=image_detect,
    window_shape=(2,2),
    pooling_type='MAX',
    strides=(2, 2),
    padding='SAME',
)
```

平均池化在很大程度上已被最大池化所取代。然而，全局平均池化仍被广泛应用于网络

```Python
model = keras.Sequential([
    pretrained_base,
    layers.GlobalAvgPool2D(),
    layers.Dense(1, activation='sigmoid'),
])
```

#### 自定义卷积

```Python
# Block Three
model = keras.Sequential([
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D()
])
```

### loss & metrics

```Python
model.compile(
    optimizer=tf.keras.optimizers.Adam(epsilon=0.01),
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)
```

### 图像增强

```Python
augment = keras.Sequential([
    # preprocessing.RandomContrast(factor=0.5),
    preprocessing.RandomFlip(mode='horizontal'), # meaning, left-to-right
    # preprocessing.RandomFlip(mode='vertical'), # meaning, top-to-bottom
    # preprocessing.RandomWidth(factor=0.15), # horizontal stretch
    # preprocessing.RandomRotation(factor=0.20),
    # preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
])
```

