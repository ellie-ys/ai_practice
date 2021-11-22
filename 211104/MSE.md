# MSE
MSE는 평균제곱오차(Mean squared error)

- ![](https://cdn-api.elice.io/api-attachment/attachment/52d2be3004e24e969f6c1869f553784a/MSE.png)


Python과 numpy를 이용해서 MSE를 직접 구현하기



### 매개변수:
* y: 정답 list
* y_hat: 비교 list
### 출력 :
* 오차값(소수점 둘째 자리에서 반올림)

### 지시사항
1. 정답 list와 비교 list의 각 요소의 차이

2. 1에서 구한 값을 제곱하세요.

3. 2에서 구한 값의 평균을 구하세요.

4. 소수점 둘째 자리에서 반올림하세요.

5. 4의 값을 반환하세요.
```
import numpy as np

def MSE(y, y_hat):
    """
    MSE 함수 구현을 채우세요.
    """
    error = y - y_hat
    squared = error **2
    mean = np.mean(squared)

    # 소수점 둘째 자리에서 반올림하세요.
    mean = round(mean, 2)
    return mean
    
# 테스트1
y     = np.array([0,   0,   1,   0,   0])
y_hat = np.array([0.1, 0.1, 0.6, 0.1, 0.1])
mse = MSE(y, y_hat)
print(mse)

# 테스트2
y     = np.array([0,   0,   1,   0,   0])
y_hat = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
mse = MSE(y, y_hat)
print(mse)

# 테스트3
y     = np.array([43,  7, 53, 86, -44])
y_hat = np.array([54, 67, 23, 96, -50])
mse = MSE(y, y_hat)
print(mse)
```
정답 데이터와 입력(추론)데이터가 얼마나 차이가 있는지 수치로 표현 할 수 있습니다. 오차함수이기 때문에 값이 작을수록 유사하고, 클수록 다릅니다.


### 참고 python
구현 - 뺄셈 
```
80 - 12 # 두 값의 차인 68이 나옵니다.
a = np.array([8, 3, -3])
b = np.array([3, 6, -6])
a - b # 각 요소의 차인 [5, -3,  3] 출력
```

구현 - 제곱
```
2 ** 3 # 2의 3제곱인 8
5 ** 2 # 5의 2제곱인 25

y = np.array([2, 3, 4])
y ** 2 # 각 값의 제곱 [4, 9, 16] 출력.
```

구현 - 평균 
```
test_list = [1, 2, 3, 4, 5]
np.mean(test_list) # 입력값의 평균인 3 출력
```