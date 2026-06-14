import numpy as np

# 예제 데이터 (측정값)
# 실제로는 CSV나 센서 데이터에서 불러오면 됩니다.
t1 = np.linspace(0, 2, 100)  # 시간
t2 = np.linspace(0, 2, 100)
a_measured1 =  -1.5*np.sin(6*t1 +0.8) + 7.2  # 예제 측정값
a_measured2 =  0.8*np.sin(3.7*t2 + 0.06) + 2.4
a_measured3 = 0*t1
'''
A1, w1, p1, B1 =-1.0625,4.0473,-1.0131, 8.8972
a_model1 = A1 * np.sin(w1*t1 - p1) + B1
A2, w2, p2, B2 = -1.7658, 6.8183, -0.8433, 8.7229
a_model2 = A2 * np.sin(w2*t2 - p2) + B2
'''
A1, w1, p1, B1 = 0.1199, 11.1134,-1.4331,0.2765
a_model1 = A1 * np.sin(w1*t1 - p1) + B1
A2, w2, p2, B2 = 0.1026, 13.7634, -0.5311, 0.2204
a_model2 = A2 * np.sin(w2*t2 - p2) + B2
# 종합손실함수 계산 함수
def total_loss(a, m):
    return np.mean(np.abs(a - m))

# 각 모델의 종합손실
loss1 = total_loss(a_model1, a_measured3)
loss2 = total_loss(a_model2, a_measured3)

print(f"모델 1 종합손실: {loss1:.4f}")
print(f"모델 2 종합손실: {loss2:.4f}")
