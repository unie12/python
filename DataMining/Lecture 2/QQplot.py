import matplotlib.pyplot as plt

data1 = [1,2,3,4,5]
data2 = [1,3,5,7,9]

max = max(data1 + data2)
min = min(data1 + data2)
baseline = range(min, max+1)

plt.plot(data1, data2, "o")
plt.plot(baseline, baseline, "r-")
plt.xlabel('data1')
plt.ylabel('data2')

plt.show()