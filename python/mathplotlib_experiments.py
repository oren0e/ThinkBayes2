import matplotlib.pyplot as plt

import numpy as np
x = np.linspace(0,5,11)
y = x ** 2

# Functional method to create matplotlib plots
plt.plot(x,y)
plt.show()

plt.plot(x,y)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Title')
plt.show()

# multiplots
plt.subplot(1,2,1)
plt.plot(x,y,'r')

plt.subplot(1,2,2)
plt.plot(y,x,'b')
plt.show()

# Object Oriented way to use matplotlib which is better
# We are going to initialize a plot object and call all the methods from it.
fig = plt.figure()
axes = fig.add_axes([0.1,0.1,0.8,0.8])  # left, bottom, width, height
axes.plot(x,y)
axes.set_xlabel('X Label')
axes.set_ylabel('Y Label')
axes.set_title('Title')
plt.show()

fig = plt.figure()  # manual method
axes1 = fig.add_axes([0.1,0.1,0.8,0.8])
axes2 = fig.add_axes([0.2,0.5,0.4,0.3])

axes1.plot(x,y)
axes1.set_title('LARGER PLOT')
axes2.plot(y,x)
axes2.set_title('SMALLER PLOT')
plt.show()

#### Part 2 ####
fig,axes = plt.subplots(nrows=1,ncols=2)  # auto method
#axes.plot(x,y)
#plt.tight_layout()
axes[1].plot(x,y)
axes[0].plot(x,y)
axes[1].set_title('Second Plot')
axes[0].set_title('First Plot')
plt.show()

# figure size and DPI
fig = plt.figure(figsize=(8,2))
ax = fig.add_axes([0,0,1,1])
ax.plot(x,y)
fig.tight_layout()
fig.show()

fig,axes = plt.subplots(nrows=2, ncols=1, figsize = (8,2))

axes[0].plot(x,y)
axes[1].plot(y,x)
plt.show()

# saving a figure
fig.savefig('my_plot.jpg',dpi=200)

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])
ax.plot(x,x**2, label='X Squared')
ax.plot(x,x**3, label = 'X Cubed')

#ax.legend(loc = 0)
ax.legend(loc = (0.1,0.1))

plt.show()

#### Part 3 ####
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])
ax.plot(x,y,color = 'purple', linewidth=3,alpha=0.5)  # RGB Hex code or strings
plt.show()

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(x,y,color = 'purple', linewidth=1, linestyle = '-', marker='o', markersize = 20,
        markerfacecolor = 'yellow', markeredgewidth = 3, markeredgecolor = 'green')  # RGB Hex code or strings
plt.show()

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(x, y, color='purple', linewidth=2, linestyle='--')

ax.set_xlim([0,1])
ax.set_ylim([0,2])
plt.show()



