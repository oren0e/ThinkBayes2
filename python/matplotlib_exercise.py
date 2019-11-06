import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0,100)
y = x*2
z = x**2

# Exercise 1
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(x,y)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('title')
plt.show()

# Exercise 2
fig = plt.figure()
ax1 = fig.add_axes([0,0,1,1])
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax2 = fig.add_axes([0.2,0.5,.2,.2])
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax1.plot(x,y)
ax2.plot(x,y)
plt.show()

# Exercise 3
fig = plt.figure()
axes1 = fig.add_axes([0,0,1,1])
axes2 = fig.add_axes([0.2,0.5,0.4,0.4])
axes1.plot(x,z)
axes1.set_xlabel('X')
axes1.set_ylabel('Z')
axes2.plot(x,y)
axes2.set_xlabel('X')
axes2.set_ylabel('Y')
axes2.set_title('zoom')
axes2.set_xlim([20,22])
axes2.set_ylim([30,50])
plt.show()

# Exercise 4
fig,axes = plt.subplots(nrows=1, ncols=2, figsize = (8,2))
axes[0].plot(x,y, color = 'blue', linestyle = '--', linewidth = 3)
axes[1].plot(x,z, color = 'red', linestyle = '-', linewidth = 3)

plt.show()