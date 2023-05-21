# DNN_Backpropagation_1
## 역전파 알고리즘

### 단순한 2층 신경망을 사용하여 XOR 문제를 학습하는 코드
활성화 함수로 시그모이드 함수 사용

### 순전파
입력 데이터 x를 첫 번째 층의 가중치 w1, 편향 b1을 더한 결과를 시그모이드 함수에 적용하여 출력 a1을 계산
두 번째 층도 반복하여 출력 y_pred를 계산

### 역전파
모델 출력 y_pred의 미분값 d_y_pred를 계산
미분값을 이용하여 역전파 과정을 통해 각 층의 가중치와 편향에 대한 미분값 계산
계산된 미분값을 사용하여 가중치와 편향 업데이트

#### 출력값
![Figure_1](https://github.com/HyunJJJUN/DNN_Backpropagation_1/assets/124676369/e870df36-5bda-445e-82b2-7a9723d11242)
![캡처](https://github.com/HyunJJJUN/DNN_Backpropagation_1/assets/124676369/f04ef076-20da-4deb-8826-de3174f31eb9)
