import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

file_path = "final1.csv"

df = pd.read_csv(file_path, skiprows=3, on_bad_lines='skip')

def clean_col(col):
    col = col.strip().lower()
    col = col.replace(" ", "_")
    col = col.replace("(m/s^2)", "")
    return col

df.columns = [clean_col(c) for c in df.columns]
print("정리된 컬럼:", df.columns.tolist())

time_col = [c for c in df.columns if 'time' in c][0]
aT_col = [c for c in df.columns if 'at' in c][0]

time = df[time_col].values
aT = df[aT_col].values

def sine_func(t, A, w, p, B):
    return A * np.sin(w * t - p) + B

# dominant frequency 추정
N = len(time)
dt = time[1] - time[0]
yf = fft(aT - np.mean(aT))  # 평균 제거
xf = fftfreq(N, dt)

xf_pos = xf[xf > 0]
yf_pos = np.abs(yf[xf > 0])

dominant_freq = xf_pos[np.argmax(yf_pos)]
w_guess = 2 * np.pi * dominant_freq

A_guess = (np.max(aT) - np.min(aT)) / 2
B_guess = np.mean(aT)
p_guess = 0
initial_guess = [A_guess, w_guess, p_guess, B_guess]

print(f"Initial guess: A={A_guess:.4f}, w={w_guess:.4f}, p={p_guess}, B={B_guess:.4f}")

popt, _ = curve_fit(sine_func, time, aT, p0=initial_guess, maxfev=5000)
A_fit, w_fit, p_fit, B_fit = popt

print(f"Fitted parameters: A={A_fit:.4f}, w={w_fit:.4f}, p={p_fit:.4f}, B={B_fit:.4f}")

plt.figure(figsize=(10,5))
plt.plot(time, aT, 'b.', label='Original Data')
plt.plot(time, sine_func(time, *popt), 'r-', label='Fitted Curve')
plt.xlabel('Time [s]')
plt.ylabel('aT [m/s^2]')
plt.legend()
plt.show()
