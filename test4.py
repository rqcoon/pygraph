from __future__ import division
from matplotlib.widgets import Slider
import numpy as np
import matplotlib.pyplot as plt
from math import pi

def equation(a,b,k):
    return k/(1+np.exp(-a*(x+b)))

#max and min variables for all sliders
k_max = 300
k_min = -300
a_max = 50
a_min = -50
b_max = 100
b_min = -100
a_init = 1
b_init = 1
k_init = 1

#create values for X and draw the plot
x = np.arange(10)
fig = plt.figure(figsize=(8,3))

#create the positions for everything
func_ax = plt.axes([.1,.3,.8,.65])
slider_ax = plt.axes([0.1, 0.15, 0.8, 0.05])
slider_ax2 = plt.axes([0.1, 0.1, 0.8, 0.05])
slider_ax3 = plt.axes([0.1, 0.05, 0.8, 0.05])

#create starting plot
#set x and y limits
plt.axes(func_ax)
plt.title('k/(1+np.exp(-a*(x+b)))')
func_plot, = plt.plot(x, k_init/(1+np.exp(-a_init*(x+b_init))), 'r')
plt.xlim(0,10)
plt.ylim(0,300)

#sliders :)
a_slider = Slider(slider_ax,
                    'a',
                    a_min,
                    a_max,
                    valinit=a_init
                    )

b_slider = Slider(slider_ax2,
                    'b',
                    b_min,
                    b_max,
                    valinit=b_init
                    )

k_slider = Slider(slider_ax3,
                    'k',
                    k_min,
                    k_max,
                    valinit=k_init
                    )

#function to change the original equation whever the sliders change
def update(a):
    func_plot.set_ydata(equation(a_slider.val, b_slider.val, k_slider.val))
    fig.canvas.draw_idle()

#update whenever the sliders change using the function above
a_slider.on_changed(update)
b_slider.on_changed(update)
k_slider.on_changed(update)

#original data for reference
ydata = np.array([1.39,11.63,20.73,39.99,72.29,125.05,150.26,169.22,231.22])
xdata = np.arange(len(ydata))
def graph(x_range):
   z = xdata
   y = ydata
   plt.plot(z,y, label='iPhone sales :)', linestyle='-',marker='o')
graph(range(0,20))

#draw everything
plt.show()
