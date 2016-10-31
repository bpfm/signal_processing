import numpy as np
import matplotlib.pyplot as plt

signal = np.loadtxt("signal.dat")

length = len(signal/2)

time = signal[:,0]
count = signal[:,1]

#time = np.zeros(shape=(length))
#count = np.zeros(shape=(length))

#for i in range(length):
#    time[i] = signal[i,0]
#    count[i] = signal[i,1]

plt.figure(1)

plt.subplot(211)
plt.plot(time,count)
plt.xlabel("Time [s]")
plt.ylabel("Count [1/s]")

fft_signal = np.fft.fft(count)

freq = np.fft.fftfreq(length)

power=(abs(fft_signal))/length

plt.subplot(212)
plt.semilogy(freq[0:500],power[0:500])
plt.semilogy(freq[501:1000],power[501:1000],color="b")
plt.xlabel("k [Hz]")
plt.ylabel("Power")

n_random=10000
n_points=1000

n_bins=500

numb=np.zeros(shape=(n_bins))
numb_vals=np.zeros(shape=(n_bins))

max_signal=5

for i in range(n_random):
    rand_signal = np.sqrt(power.real[0])*np.random.normal(0,1,(n_points))+power[0]
    fft_rand_signal=np.fft.fft(rand_signal)
    power_rand_signal=abs(fft_rand_signal)/length
    power_rand_signal[0]=0.0
    for j in range(n_points):
        numb_bin =int((power_rand_signal[j]/max_signal)*float(n_bins))
        numb[numb_bin]=numb[numb_bin]+1

sum_numb = np.sum(numb)
        
for i in range(n_bins):
    numb_vals[i]=max_signal*float(i)/float(n_bins)
    numb[i]=float(numb[i])/sum_numb

plt.figure(2)
plt.subplot(211)
plt.plot(numb_vals,numb)
plt.xlabel("Power")
plt.ylabel("Probability")

cum_numb=np.zeros(shape=(n_bins))

found0=0
found1=0
found2=0
found3=0

for i in range(n_bins):
    cum_numb[i]=np.sum(numb[0:i])
    if cum_numb[i]>=0.9995 and found0==0:
        found0=1
        print "95% confidence level is", (numb_vals[i-1]+numb_vals[i])/2.0
        ninefive_conf=(numb_vals[i-1]+numb_vals[i])/2.0
    if cum_numb[i]>=0.16 and found1==0:
        found1=1
        print "lower 1-sigma limit is", numb_vals[i]
    if cum_numb[i]>=0.84 and found2==0:
        found2=1
        print "upper 1-sigma limit is", numb_vals[i]
    if cum_numb[i]>=0.50 and found3==0:
        found3=1
        print "mean value is", numb_vals[i]

            
plt.subplot(212)
plt.plot(numb_vals,cum_numb)
plt.xlabel("Power")
plt.ylabel("Cumulative Probability")

x = np.linspace(-1,max_signal)
y = (x/x)*0.9995

y3 = np.linspace(-1,1)
x3 = (y/y)*ninefive_conf
    
plt.figure(3)

plt.subplot(211)
plt.plot(numb_vals,cum_numb)
plt.plot(x,y)
plt.plot(x3,y3,color="g")
plt.xlim(2,3)
plt.ylim(0.995,1.0)
plt.xlabel("Power")
plt.ylabel("Cumulative Probability")
y5 = np.linspace(-1,1)

plt.subplot(212)
plt.semilogy(freq[0:500],power[0:500])
plt.semilogy(freq[501:1000],power[501:1000],color="b")
plt.plot(y5,x3)
plt.xlabel("k [Hz]")
plt.ylabel("Power")
plt.xlim(-0.5,0.5)

plt.show()

