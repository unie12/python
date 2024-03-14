import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

y1=[1,2.3,8.0,2.5]
y2=[1.5, 1.7, 2.2, 2.9]
x=[1,2,3,8]
fig, (ax,ax2,ax3) = plt.subplots(1, 3, sharey=False)

ax.plot(x, y1, 'r-', x,y2,'b-')
ax2.plot(x, y1, 'r-', x,y2,'b-')
ax3.plot(x, y1, 'r-', x,y2,'b-')

ax.set_xlim([ x[0], x[1]])
ax2.set_xlim([ x[1], x[2]])
ax3.set_xlim([ x[2], x[3]])

for axx,xx in zip([ax,ax2,ax3], x[:-1]):
    axx.xaxis.set_major_locator(ticker.FixedLocator([xx]))
ax3.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))

for tick in ax3.yaxis.get_major_ticks():
    tick.label2On=True

plt.subplots_adjust(wspace=0)
plt.show()
