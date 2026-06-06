
import matplotlib.pylab as plt
import numpy as np

# features: [size_sqm, num_rooms, age_years]
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
def cost(w,b,x):
    
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
def split(X:list[float],Y:list[float],seed,percent)->tuple:
    x_test=[]
    y_test=[]
    x_train=X
    y_train=Y
    rounds=int((percent*len(X))/100)
    np.random.seed(seed)
    for i in range(len(X)-rounds):
       index= np.random.randint(0, len(x_train) - 1)
       x_test.append(x_train[index])
       y_test.append(y_train[index])
       x_train = np.delete(x_train, index, axis=0)
       y_train = np.delete(y_train, index, axis=0)

    return (x_train,y_train,x_test,y_test)







def sigmoid(z):
    return 1/(1+np.exp(-z))


def cost_for_logitstic(w,b,X,Y):
    m=len(Y)
    probs=sigmoid(X @ w + b )
    cost=-(1/m) * np.sum(Y * np.log(probs)+(1-Y) * np.log(1-probs))
    return cost

def gradient_logisitci(w,b,X,Y):
    m=len(Y)
    probs = sigmoid(X @ w + b)
    errors = probs - Y
    
    dw = (1/m) * X.T @ errors
    
    db = (1/m) * np.sum(errors)

    return dw, db


def gradienet_descent_logisitic(alpha,w,b,X,Y):
    cost_history = []
    m,n=X.shape
    while True:
        dw, db = gradient_logisitci(w, b,X,Y)
        new_w = w - alpha * dw
        new_b = b - alpha * db

        cost_history.append(cost_for_logitstic(w, b,X,Y))

        if np.all(np.abs(new_w - w) < 1e-6) and abs(new_b - b) < 1e-6:
            break

        w, b = new_w, new_b
    
    return w, b, cost_history



def cost(w, b,X,Y):
    m=len(Y)
    errors = X @ w + b - Y
    return (1 / (2 * m)) * np.sum(errors ** 2)

def gradient(w, b,X,Y,m):
    errors = X @ w + b - Y
    dw = (1 / m) * X.T @ errors
    db = (1 / m) * np.sum(errors)
    return dw, db

def gradient_descent(alpha, w, b,X,Y):
    cost_history = []
    m,n=X.shape
    while True:
        dw, db = gradient(w, b,X,Y,m)
        new_w = w - alpha * dw
        new_b = b - alpha * db

        cost_history.append(cost(w, b,X,Y))

        if np.all(np.abs(new_w - w) < 1e-6) and abs(new_b - b) < 1e-6:
            break

        w, b = new_w, new_b
    
    return w, b, cost_history




class LinearRegression():

    def __init__(self,alpha=0.01):
        self.alpha=alpha
        self.W=None
        self.B=None
        self.x_mean=0
        self.x_std=0
        self.History=None

    @staticmethod
    def split(X:list[float],Y:list[float],seed,percent)->tuple:
        x_test=[]
        y_test=[]
        x_train=X
        y_train=Y
        rounds=int((percent*len(X))/100)
        np.random.seed(seed)
        for i in range(len(X)-rounds):
            index= np.random.randint(0, len(x_train) - 1)
            x_test.append(x_train[index])
            y_test.append(y_train[index])
            x_train = np.delete(x_train, index, axis=0)
            y_train = np.delete(y_train, index, axis=0)

        return (x_train,y_train,x_test,y_test)
    
    def fit(self,X_train,Y_train)->None:
        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0)
        X_norm = (X_train - X_mean) / X_std
        m, n = X_norm.shape
        w = np.zeros(n)
        b = 0
        new_w,new_b,history=gradient_descent(self.alpha,w,b,X_norm,Y_train)
        self.History=history
        self.W=new_w
        self.B=new_b
        self.x_mean=X_mean
        self.x_std=X_std
        return
    
    
    def predict(self,X_test):
        X_norm = (X_test - self.x_mean) / self.x_std  
        return X_norm @ self.W + self.B

                
    
    def score(self, x_test, y_test):
        y_test = np.array(y_test)

        y_mean = y_test.mean()

        ss_res = np.sum((y_test - self.predict(x_test)) ** 2)
        ss_tot = np.sum((y_test - y_mean) ** 2)

        return 1 - ss_res / ss_tot
        
    def Graph(self,X_test,Y_test):
        plt.figure(1)

        plt.plot(self.History)
        plt.title("Cost over iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")

        
        plt.figure(2)
        plt.scatter(range(len(Y_test)), Y_test, label='Actual', color='blue')
        plt.scatter(range(len(Y_test)), self.predict(X_test), label='Predicted', color='red')
        plt.title("Actual vs Predicted")
        plt.legend()

        plt.show()        




class LogisticRegression():

    def __init__(self,alpha=0.01):
        self.W=None
        self.B=None
        self.alpha=alpha
        self.x_mean=None
        self.x_std=None
        self.History=None       

    @staticmethod
    def split(X:list[float],Y:list[float],seed,percent)->tuple:
        x_test=[]
        y_test=[]
        x_train=X
        y_train=Y
        rounds=int((percent*len(X))/100)
        np.random.seed(seed)
        for i in range(len(X)-rounds):
            index= np.random.randint(0, len(x_train) - 1)
            x_test.append(x_train[index])
            y_test.append(y_train[index])
            x_train = np.delete(x_train, index, axis=0)
            y_train = np.delete(y_train, index, axis=0)

        return (x_train,y_train,x_test,y_test)
    
    
    def fit(self,X_train,Y_train)->None:
        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0)
        X_norm = (X_train - X_mean) / X_std
        m, n = X_norm.shape
        w = np.zeros(n)
        b = 0
        new_w,new_b,history=gradienet_descent_logisitic(self.alpha,w,b,X_norm,Y_train)
        self.History=history
        self.W=new_w
        self.B=new_b
        self.x_mean=X_mean
        self.x_std=X_std
        return
    
    
    def predict_prob(self,X_test):
        X_norm = (X_test - self.x_mean) / self.x_std  
        return sigmoid(X_norm @ self.W + self.B)
    
    
    def predict(self,X_test):
        probs = self.predict_prob(X_test)
        
        return (probs >= 0.5).astype(int) 

    
    def score(self,X_test,Y_test):
        prediction=self.predict(X_test)
        m=len(Y_test)
        list=[]
        for i in range(m):
            if prediction[i]==Y_test[i]:
                list.append(int(1))
            else:
                list.append(int(0))
        return sum(list)/m

    
    def Graph(self,X_test,Y_test):
        plt.figure(1)

        plt.plot(self.History)
        plt.title("Cost over iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")

        
        plt.figure(2)
        plt.scatter(range(len(Y_test)), Y_test, label='Actual', color='blue')
        plt.scatter(range(len(Y_test)), self.predict(X_test), label='Predicted', color='red')
        plt.title("Actual vs Predicted")
        plt.legend()

        plt.show()        





        

         


