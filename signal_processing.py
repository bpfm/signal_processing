import numpy as np
import matplotlib.pyplot as plt

signal = np.loadtxt("signal.dat")

length = len(signal/2)

time = np.zeros(shape=(length))
count = np.zeros(shape=(length))

for i in range(length):
    time[i] = signal[i,0]
    count[i] = signal[i,1]

plt.figure(1)

plt.subplot(211)
plt.plot(time,count)
plt.xlabel("Time [s]")
plt.ylabel("Count [1/s]")


fft_signal = np.fft.fft(count)

freq = np.fft.fftfreq(length)

#print freq

fft_signal=fft_signal+1

y = np.log10(fft_signal)

plt.subplot(212)
plt.plot(freq,y.real)
plt.xlabel("Frequency [1/s]")
plt.ylabel("Counts [1/s]")

plt.show()
