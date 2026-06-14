import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

file_path = "tgyro4.csv"

# CSV 읽기
df = pd.read_csv(file_path, skiprows=3)

# 컬럼명 정리
df.columns = [c.strip() for c in df.columns]

print(df.columns)

time = df["time"].to_numpy()
omega = df["ω_total (rad/s)"].to_numpy()

# NaN 제거
mask = ~(np.isnan(time) | np.isnan(omega))
time = time[mask]
omega = omega[mask]

# 사인 함수
def sine_func(t, A, w, p, B):
    return A * np.sin(w*t - p) + B

# FFT
dt = np.mean(np.diff(time))
N = len(time)

yf = fft(omega - np.mean(omega))
xf = fftfreq(N, dt)

mask_freq = xf > 0

xf_pos = xf[mask_freq]
yf_pos = np.abs(yf[mask_freq])

dominant_freq = xf_pos[np.argmax(yf_pos)]
w_guess = 2*np.pi*dominant_freq

# 초기 추정
A_guess = (np.max(omega)-np.min(omega))/2
B_guess = np.mean(omega)

initial_guess = [A_guess, w_guess, 0, B_guess]

print("Initial guess:", initial_guess)

# 피팅
popt, _ = curve_fit(
    sine_func,
    time,
    omega,
    p0=initial_guess,
    maxfev=10000
)

A_fit, w_fit, p_fit, B_fit = popt

print(f"A = {A_fit:.4f}")
print(f"ω = {w_fit:.4f}")
print(f"f = {w_fit/(2*np.pi):.4f} Hz")
print(f"φ = {p_fit:.4f}")
print(f"B = {B_fit:.4f}")

# 그래프
plt.figure(figsize=(10,5))
plt.plot(time, omega, '.', markersize=2, label="Data")
plt.plot(
    time,
    sine_func(time,*popt),
    'r',
    linewidth=2,
    label="Sine Fit"
)

plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (rad/s)")
plt.legend()
plt.grid()
plt.show()