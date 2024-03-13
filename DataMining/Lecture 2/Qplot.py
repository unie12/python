import matplotlib.pyplot as plt
import numpy as np

data = [1,2,3,4,5,6,7,8,9,11]
data = np.asarray(data)

n = len(data)
x = np.asarray(range(n)) / (n-1)

plt.xlabel('f-value')
plt.plot(x, data, "o")
plt.show()