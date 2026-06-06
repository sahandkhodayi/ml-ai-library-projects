from Models import LinearRegression, LogisticRegression
import numpy as np

# =====================================

# LINEAR REGRESSION DEMO

# =====================================

np.random.seed(42)

size = np.random.randint(50, 200, 100)
rooms = np.random.randint(1, 8, 100)
age = np.random.randint(0, 30, 100)

X = np.column_stack((size, rooms, age))

y = (
size * 2.5 +
rooms * 20 -
age * 1.2 +
np.random.normal(0, 15, 100)
)

x_train, y_train, x_test, y_test = LinearRegression.split(
X, y, seed=42, percent=80
)

model = LinearRegression(alpha=0.01)

model.fit(x_train, y_train)

print("========== LINEAR REGRESSION ==========")
print("R² Score:", model.score(x_test, y_test))
print()

predictions = model.predict(x_test)

for actual, pred in zip(y_test[:10], predictions[:10]):
    print(f"Actual: {actual:.2f} | Predicted: {pred:.2f}")

model.Graph(x_test, y_test)

# =====================================

# LOGISTIC REGRESSION DEMO

# =====================================

salary = np.random.randint(20, 120, 200)
credit = np.random.randint(300, 850, 200)

X = np.column_stack((salary, credit))

y = (
(salary > 60) &
(credit > 600)
).astype(int)

x_train, y_train, x_test, y_test = LogisticRegression.split(
X, y, seed=42, percent=80
)

classifier = LogisticRegression(alpha=0.01)

classifier.fit(x_train, y_train)

print()
print("========== LOGISTIC REGRESSION ==========")
print("Accuracy:", classifier.score(x_test, y_test))
print()

predictions = classifier.predict(x_test)

for actual, pred in zip(y_test[:20], predictions[:20]):
    print(
    f"Actual: {actual} | Predicted: {pred}"
    )

classifier.Graph(x_test, y_test)
