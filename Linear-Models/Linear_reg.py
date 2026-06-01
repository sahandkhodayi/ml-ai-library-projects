
import matplotlib.pylab as plt
import numpy as np

# Features: [size_sqm, num_rooms, age_years]
X = np.array([
    [50,  2, 10],
    [60,  2, 15],
    [75,  3, 5],
    [80,  3, 20],
    [90,  4, 8],
    [100, 4, 3],
    [110, 4, 12],
    [120, 5, 7],
    [130, 5, 2],
    [150, 6, 1]
])

# Price in thousands
y = np.array([150, 165, 210, 195, 270, 310, 280, 350, 390, 450])
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


# Normalize
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_norm = (X - X_mean) / X_std

m, n = X_norm.shape
w = np.zeros(n)
b = 0

def cost(w, b):
    errors = X_norm @ w + b - y
    return (1 / (2 * m)) * np.sum(errors ** 2)

def gradient(w, b):
    errors = X_norm @ w + b - y
    dw = (1 / m) * X_norm.T @ errors
    db = (1 / m) * np.sum(errors)
    return dw, db

def gradient_descent(alpha, w, b):
    cost_history = []
    
    while True:
        dw, db = gradient(w, b)
        new_w = w - alpha * dw
        new_b = b - alpha * db

        cost_history.append(cost(w, b))

        if np.all(np.abs(new_w - w) < 1e-6) and abs(new_b - b) < 1e-6:
            break

        w, b = new_w, new_b
    
    return w, b, cost_history

final_w, final_b, cost_history = gradient_descent(0.01, w, b)

print("Weights:", final_w)
print("Bias:", final_b)

predictions = X_norm @ final_w + final_b

# Plot 1 - cost going down
plt.figure(1)
plt.plot(cost_history)
plt.title("Cost over iterations")
plt.xlabel("Iteration")
plt.ylabel("Cost")

# Plot 2 - actual vs predicted
plt.figure(2)
plt.scatter(range(len(y)), y, label='Actual', color='blue')
plt.scatter(range(len(y)), predictions, label='Predicted', color='red')
plt.title("Actual vs Predicted")
plt.legend()

plt.show()