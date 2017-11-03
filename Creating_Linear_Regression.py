#Implementing,Train and Test Linear Regression Model
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import  style
import random
style.use('fivethirtyeight')

def create_dataset(hm, variance, step=2, correlation=False):    #Creating Raandom Dataset 
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)         
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val = val - step
    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)   


xs,ys = create_dataset(40, 10, 2, 'neg')





def best_fit_slope_and_intercept(xs,ys):                              #Slope of a line and Y intercept
    m = (  ( (mean(xs) * mean(ys))  -  mean(xs * ys) )   /
           ( (mean(xs) * mean(xs)) -  mean(xs*xs) ) )
    b = mean(ys) - m * mean(xs)
    return m , b

def squared_error(ys_orig,ys_line):             # finding squared error
    return sum((ys_line-ys_orig)**2)

def coefcnt_of_determnation(ys_orig,ys_line):               #finding R^2 value  =  1- (SE Y hat/ SE Ymean)
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig,ys_line)
    squared_error_y_mean = squared_error(ys_orig,y_mean_line)
    return (1 - (squared_error_regr/ squared_error_y_mean))

m,b = best_fit_slope_and_intercept(xs,ys)

regression_line = [(m *x)+b for x in xs]
predict_x = 5
predict_y = (m*predict_x)+b

r_squared = coefcnt_of_determnation(ys, regression_line)
print(r_squared)

plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y,color = 'r',s=100)
plt.plot(xs, regression_line)
plt.show()
