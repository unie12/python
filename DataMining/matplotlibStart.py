import matplotlib.pyplot as plt

x=[1,2,3,4]
y=[3,6,9,12]

# plt.plot(x,y)
# plt.xlim(0, 10)
# plt.ylim(2, 15)
# plt.show()


# plt.plot(x, y)
# plt.xticks([0,2,4,6])
# plt.yticks([2,4,6,8,10,12])
# plt.show()

x1=[1,2,3,4]
y1=[3,6,9,12]
plt.plot(x1,y1,'r', label='Line1')

x2=[1,2,3,4]
y2=[2,4,6,8]
plt.plot(x2,y2, label='line2')
plt.show()

# plt.legend(loc='upper Left')
plt.xlabel('x-value')
plt.ylabel('y-value')
plt.title('simple plot')
plt.show()