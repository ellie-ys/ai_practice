#CNN 구현

MNIST 데이터셋
MNIST 데이터셋은 아래와 같이 숫자 0부터 9까지의 수를 손으로 쓴 28 × 28의 이진 이미지 데이터셋

60,000개의 학습 셋과 10,000개의 테스트셋이 있습니다.
![](https://cdn-api.elice.io/api-attachment/attachment/88e1a1e6ce7c42478bf7af80badb1e19/image.png)

우리는 이 데이터셋으로 우리만의 CNN을 학습 시켜 숫자 이미지를 분리하는 분류기를 만들 것이다.

케라스에는 모델을 쉽게 구현할 수 있는 models, layers와 활성화 함수들이 있는 activations 모듈들이 있다
```
from tensorflow.keras import layers, models, activations
```
모델 생성 예시
```
# 순차적으로 레이어가 쌓이는 모델 만들기
model = models.Sequential()

# 컨볼루션 레이어 만들기
model.add(layers.Convolution2D(32, (3, 3), activation=activations.relu, input_shape=(28, 28, 1)))

# 풀링 적용하기
model.add(layers.MaxPooling2D((2, 2)))

# 1차원  텐서로 변환하기
model.add(layers.Flatten())

# FC레이어 만들기
model.add(layers.Dense(64, activation='relu'))

# 모델 구조 출력하기
model.summary()
```
- 지시사항 
간단한 CNN 구조 따라 만들기

![](https://elice-api-cdn.azureedge.net/api-attachment/attachment/e165b846a08e410598f8bb5ce9ce94db/image.png)

1. keras의 models, layers, activations 모듈을 활용하여 위의 구조를 갖는 CNN을 만들고 모델의 구조를 summary()함수를 통해 출력
2. 모든 Convolution 레이어의 activation함수는 relu를 사용합니다.

3. 마지막에 Dense Layer는 첫 번째 레이어는 relu를 두 번째 레이어는 softmax를 쓰도록 합니다.

4. Convolution 레이어는 패딩을 주지 않습니다.

5. Convolution 레이어는 스트라이드를 주지 않습니다.

6. 모든 Convolution 레이어의 커널 사이즈는 (3, 3)으로 합니다.


```
from tensorflow.keras import datasets, layers, models, activations


# 모델 변수를 선언합니다.
model = models.Sequential()

# 모델에 첫 번째 입력 레이어를 추가합니다.
model.add(layers.Convolution2D(32, (3, 3), activation=activations.relu, input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

# 아래에 지시상항에 있는 모델 구조가 되도록 나머지 모델 구조를 선언해주세요.

#한세트로 쓰인다. convol, maxpooling
model.add(layers.Convolution2D(64,(3,3), activation = activations.relu))
model.add(layers.MaxPooling2D(pool_size=(2,2))) #가장먼저 입력받아야하기떄문에 새략가능한 poolsize

model.add(layers.Convolution2D(64,(3,3), activation = activations.relu))


model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10, activation = 'softmax'))
# Model 구조를 출력합니다.
model.summary()
```

- cnn training 폴더에 구현된 모델 
![](https://www.notion.so/ysyschoi/cnnc-73bfd2405e0843d7ac40e1cabb63c44d#1a2c4caed1e04121b5f0362f90bda429)