import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# CSV 파일 경로
file_path = "curve3.csv"

# CSV 읽기: 상단 3줄 무시, 문제 있는 행 무시
df = pd.read_csv(file_path, skiprows=3, on_bad_lines='skip')

# 컬럼 이름 정리: 소문자화, 공백 제거, 괄호 제거
def clean_col(col):
    col = col.strip().lower()
    col = col.replace(" ", "_")
    col = col.replace("(m/s^2)", "")
    return col

df.columns = [clean_col(c) for c in df.columns]
print("정리된 컬럼:", df.columns.tolist())

# 실제 컬럼 선택
time_col = [c for c in df.columns if 'time' in c][0]
aT_col = [c for c in df.columns if 'at' in c][0]

time = df[time_col].values
aT = df[aT_col].values

# 사인함수 정의
def sine_func(t, A, w, p, B):
    return A * np.sin(w * t - p) + B

# FFT로 dominant frequency 추정
N = len(time)
dt = time[1] - time[0]
yf = fft(aT - np.mean(aT))  # 평균 제거
xf = fftfreq(N, dt)

# 양수 주파수만
xf_pos = xf[xf > 0]
yf_pos = np.abs(yf[xf > 0])

dominant_freq = xf_pos[np.argmax(yf_pos)]
w_guess = 2 * np.pi * dominant_freq

# 초기 추정값
A_guess = (np.max(aT) - np.min(aT)) / 2
B_guess = np.mean(aT)
p_guess = 0
initial_guess = [A_guess, w_guess, p_guess, B_guess]

print(f"Initial guess: A={A_guess:.4f}, w={w_guess:.4f}, p={p_guess}, B={B_guess:.4f}")

# curve_fit 적용 (maxfev 증가)
popt, _ = curve_fit(sine_func, time, aT, p0=initial_guess, maxfev=5000)
A_fit, w_fit, p_fit, B_fit = popt

print(f"Fitted parameters: A={A_fit:.4f}, w={w_fit:.4f}, p={p_fit:.4f}, B={B_fit:.4f}")

# 시각화
plt.figure(figsize=(10,5))
plt.plot(time, aT, 'b.', label='Original Data')
plt.plot(time, sine_func(time, *popt), 'r-', label='Fitted Curve')
plt.xlabel('Time [s]')
plt.ylabel('aT [m/s^2]')
plt.legend()
plt.show()
