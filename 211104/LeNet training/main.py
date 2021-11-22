import os 
import cv2
import numpy
from tensorflow.keras import datasets, layers, models, activations, losses, optimizers, metrics

import tensorflow as tf
import numpy as np


# mnist 데이터 셋을 로드합니다.
# 각각 학습셋(이미지, 라벨), 테스트 셋(이미지, 라벨)으로 구성이 되어 있습니다.
data_path = os.path.abspath("./mnist.npz")
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data(path=data_path)

train_cnt, test_cnt = 60000, 10000
train_images, train_labels = train_images[:train_cnt], train_labels[:train_cnt]
test_images, test_labels = test_images[:test_cnt], test_labels[:test_cnt]

# 학습 셋은 60000개의 28x28 이진 이미지이므로 reshaping을 해줍니다.
train_images = train_images.reshape((train_cnt, 28, 28, 1))

# 테스트 셋은 10000개의 28x28 이진 이미지이므로 reshaping을 해줍니다.
test_images = test_images.reshape((test_cnt, 28, 28, 1))

# LeNet의 입력은 32x32 이미지 입니다. 패딩을 주어서 28 x 28에서 32 x 32 이미지로 만듭니다.
train_images = numpy.pad(train_images, [[0, 0], [2,2], [2,2], [0,0]], 'constant')
test_images = numpy.pad(test_images, [[0, 0], [2,2], [2,2], [0,0]], 'constant')
print('train_images :', train_images.shape, type(train_images))
print('test_images :', test_images.shape, type(test_images))

# 픽셀 값을 0~1 사이로 정규화합니다.
train_images, test_images = train_images / 255.0, test_images / 255.0

# 모델을 구조를 선언합니다.
model = models.Sequential()

model.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(32, 32, 1)))

model.add(layers.AveragePooling2D((2,2), strides = (2,2)))

model.add(layers.Conv2D(16, (5,5), strides=(1,1), activation= 'tanh'))
model.add(layers.AveragePooling2D((2,2), strides = (2,2)))
model.add(layers.Conv2D(120, (5,5), strides=(1,1), activation= 'tanh'))


model.add(layers.Flatten())
model.add(layers.Dense(84, 'tanh'))
model.add(layers.Dense(10, 'softmax'))


# 모델을 컴파일 합니다.
adam_optimizer = optimizers.Adam()
loss_function = losses.sparse_categorical_crossentropy
metric = metrics.categorical_accuracy


model.compile(optimizer = adam_optimizer, loss = loss_function, matrics = [metric])

# 모델을 학습데이터로 학습합니다.
model.fit(train_images, train_labels, epochs =1)

# 모델을 평가합니다.

test_loss, test_acc = model.evaluate(test_images, test_labels)


# 학습 결과를 출력합니다.
print("test_loss:", test_loss, "test_acc:", test_acc)

# 모델에 테스트 이미지를 넣고 예측값을 확인해봅니다.
test_img = cv2.imread("2.png", cv2.IMREAD_GRAYSCALE)
print(test_img)

# 입력 이미지의 픽셀을 0~1 사이로 정규화 합니다.
test_img = test_img / 255.0
row, col, channel = test_img.shape[0], test_img.shape[1], 1
confidence = model.predict(test_img.reshape((1, row, col, channel)))

for i in range(confidence.shape[1]):
    print(f"{i} 일 확률 = {confidence[0][i]}")

print(f"정답은 : {numpy.argmax(confidence, axis=1)}")
