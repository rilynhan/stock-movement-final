import pandas as pd
from scipy.stats import linregress
from scipy.interpolate import UnivariateSpline
import numpy as np
import matplotlib.pyplot as plt

def linear_regression_slope(prices):
    n = len(prices)
    x = np.arange(n)
    y = np.array(prices)
    res = linregress(x, y)
    # print(res)
    return res.slope

def center_moving_average(prices, half_window_size):
    smoothed_slope = []

    for i in range(half_window_size, len(prices)-half_window_size):
        smoothed_slope.append(linear_regression_slope(prices[i-half_window_size:i+half_window_size+1]))
    smoothed_slope = [None] * half_window_size + smoothed_slope + [None] * half_window_size
    return smoothed_slope

def draw_linear_with_slope(x, price, slope):
    plt.plot([x-0.5, x+0.5], [price-0.5*slope, price+0.5*slope], linewidth=2, color='cyan')

half_window_size = 3
df = pd.read_csv('./stock_data/300014.csv')
prices = df['close'].to_list()
x = np.arange(len(prices))
smoothed_slope = center_moving_average(prices, half_window_size)
spline = UnivariateSpline(x, prices)
x_interpolate = []
for i in x:
    x_interpolate += [i+0.1*j for j in range(10)]
prices_interpolate = spline(x_interpolate)

plt.figure(figsize=(200, 20))
plt.plot(x_interpolate, prices_interpolate, linewidth=2, color='red', zorder=1)
# for i in range(half_window_size, len(x)-half_window_size):
    # draw_linear_with_slope(i, prices[i], smoothed_slope[i])
# for i in x:
#     draw_linear_with_slope(i, prices[i], spline.derivative(n=1)(i))
plt.scatter(x, prices, s=10, marker='o', zorder=2)
# plt.plot(prices, label='origin price', marker='o', markersize=3)

plt.savefig(f'temp_{len(x)}.png')