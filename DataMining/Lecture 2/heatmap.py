import matplotlib.pyplot as plt
import numpy as np

data = np.random.random((16, 16))
plt.imshow(data, cmap='PuBu',
           interpolation='nearest')
plt.show()