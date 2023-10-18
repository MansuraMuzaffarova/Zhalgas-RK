# Zhalgas-RK
import pandas as pd
from sklearn.model_selection import train_test_split
from tslearn.svm import TimeSeriesSVC
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import matplotlib.pyplot as plt
import seaborn as sns

try:
    data = pd.read_csv("D:/blood1.csv")
except FileNotFoundError:
    print("Файл не найден.")
    exit(1)

X = data.drop(columns=["BloodGlucoseLevel"])
y = data["BloodGlucoseLevel"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
clf = TimeSeriesSVC(kernel="rbf")
losses = []
start_time = time.time()
clf.fit(X_train, y_train)
end_time = time.time()
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
for epoch in range(20):
    start_time = time.time()
    clf.fit(X_train, y_train)
    end_time = time.time()
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    losses.append(accuracy)
plt.figure(figsize=(8, 6))
plt.plot(range(1, 21), losses, marker='o', linestyle='-')
plt.title('График обучения TimeSeriesSVC')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.grid(True)
plt.show()
# Вывод матрицы ошибок
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Матрица ошибок')
plt.xlabel('Предсказанные классы')
plt.ylabel('Фактические классы')
plt.show()

print("Точность TimeSeriesSVC:", accuracy)
print("Время обучения модели:", end_time - start_time, "секунд")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, c='green', label='Predicted', marker='o')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2, label='Ideal', linestyle='--')
plt.title('Результаты TimeSeriesSVC для BloodGlucoseLevel')
plt.xlabel('Фактические значения')
plt.ylabel('Предсказанные значения')
plt.legend()
plt.grid(True)
plt.show()
