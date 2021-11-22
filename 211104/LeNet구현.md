# LeNet
CNN을 가장 처음 도입한 적용한 모델

LeNet의 경우 LeNet-1부터 LeNet-5까지 다양한 버전으로 존재한다


- LeNet의 구조
- ![](https://elice-api-cdn.azureedge.net/api-attachment/attachment/6375acd884d74f8ab3cab9dc804d701a/image.png)

- LeNet-5를 직접 구현
```
import logging, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.backend.tensorflow_backend import tf
logger = tf.get_logger()
logger.setLevel(logging.FATAL)

import keras
from tensorflow.keras import datasets, layers, models, activations, utils

# 모델 변수를 선언합니다.
model = models.Sequential()
# 모델에 첫번째 입력 레이어 추가
model.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(32, 32, 1)))


model.add(layers.AveragePooling2D((2,2), strides = (2,2)))
model.add(layers.Conv2D(16, (5,5), strides=(1,1), activation= 'tanh'))
model.add(layers.AveragePooling2D((2,2), strides = (2,2)))
model.add(layers.Conv2D(120, (5,5), strides=(1,1), activation= 'tanh'))


model.add(layers.Flatten())
model.add(layers.Dense(84, 'tanh'))
model.add(layers.Dense(10, 'relu'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer='SGD')

# Model 구조 확인
model.summary()
```
- LeNet의 각 레이어별 커널, 스트라이드 사이즈와 사용되는 활성화 함수가 적인 표
- ![](https://elice-api-cdn.azureedge.net/api-attachment/attachment/1d3c8228eb904e6cb151e55a70d264ce/image.png)

- CNN과 어떤 점이 다른지?
- CNN 과 같은 데이터셋, 같은 손실 함수, 같은 평가 지표, 같은 epoch, 같은 optimizer를 사용하여 LeNet을 학습 시켰을때 loss와 accuracy를 비교해보고 어떤것들이 성능의 차이를 가지고 오는지 생각해보기