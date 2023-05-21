import numpy as np
import matplotlib.pyplot as plt

# 활성화 함수 (시그모이드)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 학습 데이터
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# 가중치 초기화
np.random.seed(0)
W1 = np.random.randn(2, 2)
b1 = np.zeros((1, 2))
W2 = np.random.randn(2, 1)
b2 = np.zeros((1, 1))

# 학습 파라미터
learning_rate = 0.1
epochs = 10000

# 손실 기록
loss_history = []

# 학습
for epoch in range(epochs):
    # 순전파
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    y_pred = sigmoid(z2)
    
    # 손실 계산
    loss = np.mean(0.5 * (y - y_pred)**2)
    loss_history.append(loss)
    
    # 역전파
    d_y_pred = (y_pred - y) * y_pred * (1 - y_pred)
    d_z2 = d_y_pred
    d_W2 = np.dot(a1.T, d_z2)
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)
    d_a1 = np.dot(d_z2, W2.T)
    d_z1 = d_a1 * a1 * (1 - a1)
    d_W1 = np.dot(X.T, d_z1)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)
    
    # 가중치 업데이트
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1
    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2

# 가중치와 편향 출력
print("W1:", W1)
print("b1:", b1)
print("W2:", W2)
print("b2:", b2)

# 학습 과정 그래프
plt.plot(range(epochs), loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
