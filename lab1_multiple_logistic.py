# Лабораторная работа. Задание 5.
# 2. Множественная логистическая регрессия
# Данные: UCI Bank Marketing, файл bank.csv
# Y = y
# Факторы: duration и pdays

# 4. Импорт необходимых пакетов и классов
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from mpl_toolkits.mplot3d import Axes3D 

# 5. Загрузка данных и их вывод
FILE = Path(__file__).with_name("bank.csv")
df = pd.read_csv(FILE, sep=";")

print("=" * 70)
print("ПЕРВЫЕ 10 СТРОК ДАННЫХ")
print("=" * 70)
print(df.head(10))
print("\nРазмер набора данных:", df.shape)
print("\nНазвания столбцов:")
print(df.columns.tolist())

# 6. Переменная отклика и факторы

# Y — y: оформил ли клиент срочный вклад.
# X1 — duration: длительность последнего контакта.
# X2 — pdays: число дней после предыдущего контакта;
#      -1 означает, что ранее клиента не контактировали.

X = df[["duration", "pdays"]]
y = df["y"].map({"no": 0, "yes": 1})

print("\nПеременная Y: y (no=0, yes=1)")
print("Факторы X1, X2: duration, pdays")


# 7. 3-мерная диаграмма рассеивания
# создаём фигуру и 3D-оси
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")
# маски для разделения классов целевой переменной (0 — "no", 1 — "yes")
mask_no = y == 0
mask_yes = y == 1
# точки класса "no" — низкая непрозрачность, т.к. их обычно больше
ax.scatter(
    X.loc[mask_no, "duration"],
    X.loc[mask_no, "pdays"],
    y.loc[mask_no],
    alpha=0.18,
    s=12,
    label="no"
)
# точки класса "yes" — более контрастные и крупные, чтобы выделялись на фоне
ax.scatter(
    X.loc[mask_yes, "duration"],
    X.loc[mask_yes, "pdays"],
    y.loc[mask_yes],
    alpha=0.45,
    s=20,
    label="yes"
)
# подписи осей и заголовок графика
ax.set_xlabel("duration, сек.")
ax.set_ylabel("pdays, дней")
ax.set_zlabel("y")
ax.set_title("3D диаграмма рассеивания")
# легенда для различения классов
ax.legend()
# автоматическая подгонка отступов, сохранение в файл и отображение
plt.tight_layout()
plt.savefig("04_scatter_3d_multiple.png", dpi=200)
plt.show()

# 8. Обучающая и тестовая выборки

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=9, stratify=y
)
print("\nРазмер обучающей выборки:", X_train.shape, y_train.shape)
print("Размер тестовой выборки:", X_test.shape, y_test.shape)


# 9. Построение натренированной модели LogisticRegression
model = LogisticRegression(max_iter=2000, random_state=9)
model.fit(X_train, y_train)

print("\nМодель обучена.")

# 10. Выбор наблюдений из тестовой выборки и прогноз
# берём первые 10 наблюдений из тестовой выборки для наглядной проверки модели
sample = X_test.head(10)
sample_true = y_test.loc[sample.index] # фактические значения y для этих наблюдений
# получаем предсказания модели: классы (0/1) и вероятности класса "yes"
sample_pred = model.predict(sample)
# собираем результаты в единую таблицу для удобного сравнения
result_sample = pd.DataFrame({
    "duration": sample["duration"],
    "pdays": sample["pdays"],
    "Фактическое y": sample_true,
    "Прогноз y": sample_pred,
    "Вероятность yes": model.predict_proba(sample)[:, 1] # столбец 1 — вероятность класса "yes"
})
# выводим таблицу целиком, без обрезки строк/столбцов
print("\nПримеры наблюдений тестовой выборки:")
print(result_sample.to_string())

# 11. Коэффициенты логистической регрессионной модели
b0 = model.intercept_[0]
b_duration = model.coef_[0][0]
b_pdays = model.coef_[0][1]

print("\nКоэффициенты модели:")
print("Свободный член b0 =", b0)
print("Коэффициент duration =", b_duration)
print("Коэффициент pdays =", b_pdays)

# 12. Прогноз переменной отклика для новых объектов
# создаём новые гипотетические наблюдения с разными комбинациями duration и pdays
new_objects = pd.DataFrame({
    "duration": [60, 150, 300, 600, 1000],
    "pdays": [-1, 30, 90, 180, 330]
})
# получаем предсказанный класс (0/1) и вероятность класса "yes" для каждого объекта
new_pred = model.predict(new_objects)
new_prob = model.predict_proba(new_objects)[:, 1]
# формируем итоговую таблицу: исходные признаки + результат в человекочитаемом виде
new_result = new_objects.copy()
new_result["Прогноз класса"] = np.where(new_pred == 1, "yes", "no")  # переводим 0/1 в "no"/"yes"
new_result["Вероятность yes"] = new_prob
# выводим таблицу без индекса, чтобы было проще читать
print("\nПрогноз для новых объектов:")
print(new_result.to_string(index=False))


# 13. Средняя абсолютная, квадратичная ошибка и качество
y_test_pred = model.predict(X_test)
# средняя абсолютная оценка ошибки
mae = mean_absolute_error(y_test, y_test_pred)
# средняя квадратичная оценка ошибки
mse = mean_squared_error(y_test, y_test_pred)
# качество
accuracy = accuracy_score(y_test, y_test_pred)

print("\nОценка модели на тестовой выборке:")
print(f"MAE = {mae:.6f}")
print(f"MSE = {mse:.6f}")
print(f"Accuracy (качество) = {accuracy:.6f} ({accuracy*100:.2f}%)")

