import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


np.random.seed(42)  
data = np.random.normal(80, 10, 1440)  

def elevated(data, threshold=150, consecutive_minutes=15):
    elevated_periods = []
    consecutivecount = 0
    for i, rate in enumerate(data):
        if rate > threshold:
            consecutive_count += 1
        else:
            if consecutive_count >= consecutive_minutes:
                elevated_periods.append((i - consecutive_count, i))
            consecutive_count = 0
    return elevated_periods

def low_pass_filter(data, cutoff=0.1, fs=1.0, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def compute_hourly_averages(data, minutes_per_hour=60):
    return np.mean(data.reshape(-1, minutes_per_hour), axis=1)




plt.figure(figsize=(12, 6))
time = np.arange(1440)  

plt.plot(time,data, label='Original Noisy Data')

smooth= low_pass_filter(data)

plt.plot(time, smooth, label='Smoothed Data', color='orange')


hourly_time = np.arange(0, 1440, 60)
hourly_avg = compute_hourly_averages(smooth)
plt.plot(hourly_time, hourly_avg, label='Hourly Averages', marker='o', color='red', linestyle='--')




plt.title('rush hour analysis')
plt.xlabel('time')
plt.ylabel('vehicle_count')
plt.legend()
plt.show()