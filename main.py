#imports (yes i comment this shit)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

#for regression later on
def exponenial_func(x, a, b, c):
    return a*np.exp(-b*x)+c

#draw graph and set line positions
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.xlabel('Time (2007 to 2015)')
plt.ylabel('iPhone sales (in millions)')

#data for graphing, finding the amount of points to graph along x
ydata = np.array([1.39,11.63,20.73,39.99,72.29,125.05,150.26,169.22,231.22])
xdata = np.arange(len(ydata))
xleng = len(ydata)
yleng = max(ydata)

trialX = np.linspace(xdata[0],xdata[-1],1000)

#create predictions using exponential regression/fitting
popt, pcov = curve_fit(exponenial_func, xdata, ydata, p0=(1, 1e-6, 1))
yy = exponenial_func(trialX, *popt)

#print equation
raw_coef = (curve_fit(lambda t,a,b: a*np.exp(b*t),  xdata,  ydata,  p0=(4, 0.1)))
coef = raw_coef[0]
print(coef[0],'exp (',coef[1],'x )')

#find highest value in data for y axis (create some overhead), use length of data for x axis
plt.ylim(0,yleng + 20)
plt.xlim(0,xleng)

def graph(x_range):
   x = xdata
   y = ydata
   plt.plot(x,y, label='original data', linestyle='',marker='o')
graph(range(0,20))

def graph2(x_range):
   t = trialX
   a = yy
   plt.plot(t,a, label='fitted curve', linestyle='--', color='purple')
graph2(range(0,20))

def graph3(x_range):
    g = xdata
    f = coef[0]*np.exp(coef[1]*g)
    plt.plot(g,f, label='fitted curve from equation')
graph3(range(0,20))

plt.legend(loc="upper left")
plt.show()
#imports (yes i comment this shit)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

#for regression later on
def exponenial_func(x, a, b, c):
    return a*np.exp(-b*x)+c

#draw graph and set line positions
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.xlabel('months (March 2012 --> September 2016)')
plt.ylabel('Apps downloaded (billions)')

#data for graphing, finding the amount of points to graph along x
ydata = np.array([25,30,35,40,50,60,75,80,100,130,140])
xdata = np.arange(len(ydata))
xleng = len(ydata)
yleng = max(ydata)

trialX = np.linspace(xdata[0],xdata[-1],1000)

#create predictions using exponential regression/fitting
popt, pcov = curve_fit(exponenial_func, xdata, ydata, p0=(1, 1e-6, 1))
yy = exponenial_func(trialX, *popt)

#print equation
raw_coef = (curve_fit(lambda t,a,b: a*np.exp(b*t),  xdata,  ydata,  p0=(4, 0.1)))
coef = raw_coef[0]
print(coef[0],'exp (',coef[1],'x )')

#find highest value in data for y axis (create some overhead), use length of data for x axis
plt.ylim(0,yleng + 20)
plt.xlim(0,xleng)

def graph(x_range):
   x = xdata
   y = ydata
   plt.plot(x,y, label='original data')
graph(range(0,20))

def graph2(x_range):
   t = trialX
   a = yy
   plt.plot(t,a, label='fitted curve')
graph2(range(0,20))

def graph3(x_range):
    g = xdata
    f = coef[0]*np.exp(coef[1]*g)
    plt.plot(g,f, label='fitted curve from equation')
graph3(range(0,20))

plt.legend(loc="upper left")
plt.show()
