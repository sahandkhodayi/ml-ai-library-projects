
import matplotlib.pylab as plt
import numpy as np

x = np.array([
1,2,3,4,5,6,7,8,9,10
])

y = np.array([
6.5,
6.8,
10.2,
9.5,
14.7,
15.1,
18.9,
17.2,
22.4,
20.3
])

def cost(w,b):
    
    sum=0
    for i in range(len(x)):
        power=w*x[i]+b-y[i]
        sum+=power**2
    return sum / (2*len(x))

# w_vals = np.linspace(0,10,50)
# b_vals = np.linspace(0,10,50)

# W, B = np.meshgrid(w_vals, b_vals)
# Z = np.zeros_like(W)

# for i in range(W.shape[0]):
#     for j in range(W.shape[1]):
#         Z[i,j] = cost(W[i,j], B[i,j])

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.plot_surface(W, B, Z, cmap="viridis")

# ax.set_xlabel("w")
# ax.set_ylabel("b")
# ax.set_zlabel("cost")



def partial_w(w,b):
    sum=0
    for i in range(len(x)):
        power=(w*x[i]+b-y[i])*x[i]
        sum+=power
    return sum / (len(x))
    
def partial_b(w,b):
    sum=0
    for i in range(len(x)):
        power=w*x[i]+b-y[i]
        sum+=power
    return sum / (len(x))

def finding_local_minima(alpha,w,b):

    while True:
        old_w=w
        old_B=b
        w=old_w-(alpha*partial_w(old_w,old_B))
        b=old_B-(alpha*partial_b(old_w,old_B))
        
        

        if abs(old_w - w) < 1e-6 and abs(old_B - b) < 1e-6:
            break
    return w,b

final_w,final_B=finding_local_minima(0.001,0,0)   



y_line = final_w*x+final_B

plt.scatter(x, y)          # data points
plt.plot(x, y_line, 'r')   # regression line

plt.show()