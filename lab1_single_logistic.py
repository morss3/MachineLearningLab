# Лабораторная работа. Задание 5.
# 1. Однофакторная логистическая регрессия
# Данные: UCI Bank Marketing, файл bank.csv
# Y = y (подписал ли клиент срочный вклад)
# X = duration (продолжительность последнего телефонного контакта, сек.)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

#5. Загрузка данных и их вывод
FILE = Path(__file__).with_name("bank.csv")
df = pd.read_csv(FILE, sep=";")

print("=" * 70)
print("ПЕРВЫЕ 10 СТРОК ДАННЫХ")
print("=" * 70)
print(df.head(10))
print("\nРазмер набора данных:", df.shape)
print("\nНазвания столбцов:")
print(df.columns.tolist())
 
# 6. Переменная отклика и фактор
# Y — y: результат оформления срочного вклада.
# X — duration: длительность последнего контакта в секундах.
X = df[["duration"]]
y = df["y"].map({"no": 0, "yes": 1})

print("\nПеременная X: duration")
print("Переменная Y: y (no=0, yes=1)")

# 7. Диаграмма рассеивания
# добавляем небольшой случайный шум (jitter) к y, чтобы точки на уровнях 0 и 1
# не накладывались друг на друга сплошной линией, а были видны как облако
jitter = np.random.normal(0, 0.025, size=len(y))
plt.figure(figsize=(10, 6))
# сама диаграмма рассеивания: duration по X, "размытый" y по Y
plt.scatter(X["duration"], y + jitter, alpha=0.3, s=15)
# возвращаем подписи оси Y к исходным категориям, несмотря на добавленный шум
plt.yticks([0, 1], ["no", "yes"])
plt.xlabel("Продолжительность последнего контакта, сек.")
plt.ylabel("Оформление вклада")
plt.title("Диаграмма рассеивания: duration и y")
plt.grid(alpha=0.2)
plt.savefig("01_scatter_single.png", dpi=200)
plt.show()
 
# 8. Обучающая и тестовая выборки 
# stratify=y гарантирует, что доля "yes" и "no" в train и test одинаковая 
# (или очень близкая), а не зависит от случайности разбиения.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=9, stratify=y
)

print("\nРазмер обучающей выборки:", X_train.shape, y_train.shape)
print("Размер тестовой выборки:", X_test.shape, y_test.shape)

# 9. Построение натренированной модели LogisticRegression
model = LogisticRegression(max_iter=1000, random_state=9)
model.fit(X_train, y_train)

print("\nМодель обучена.")

# 10. Выбор наблюдений из тестовой выборки и прогноз
# берём первые 10 объектов тестовой выборки для демонстрации работы модели
sample = X_test.head(10)
sample_true = y_test.loc[sample.index]  # реальные метки для этих наблюдений
# предсказанный класс модели
sample_pred = model.predict(sample)
# собираем всё в одну таблицу: признак, факт, прогноз и вероятность
result_sample = pd.DataFrame({
    "duration": sample["duration"],
    "Фактическое y": sample_true,
    "Прогноз y": sample_pred,
    "Вероятность yes": model.predict_proba(sample)[:, 1]  # вероятность класса "yes"
})
print("\nПримеры наблюдений тестовой выборки:")
print(result_sample.to_string())
# краткое текстовое пояснение к таблице
print("\nВывод: модель для каждого наблюдения определяет класс 0 (no) или 1 (yes).")
print("Дополнительно последний столбец показывает вероятность оформления вклада.")

# 11. Коэффициенты логистической регрессионной модели
b0 = model.intercept_[0]
b1 = model.coef_[0][0]

print("\nКоэффициенты модели:")
print("Свободный член b0 =", b0)
print("Коэффициент b1 =", b1)

# 12. Логистическая кривая на диаграмме рассеивания
# создаём равномерную сетку значений duration в диапазоне от min до max,
# чтобы построить гладкую кривую предсказанных вероятностей
x_curve = np.linspace(X["duration"].min(), X["duration"].max(), 500)
# считаем вероятность класса "yes" для каждой точки сетки (однофакторная модель по duration)
p_curve = model.predict_proba(pd.DataFrame({"duration": x_curve}))[:, 1]
plt.figure(figsize=(10, 6))
# фактические наблюдения с jitter (как и раньше) — фон для сравнения с моделью
plt.scatter(X["duration"], y + jitter, alpha=0.18, s=12, label="Наблюдения")
# сама логистическая кривая — визуализация того, как модель оценивает вероятность yes
plt.plot(x_curve, p_curve, linewidth=3, label="Логистическая кривая", color="red")
plt.yticks([0, 1], ["no", "yes"])
plt.xlabel("Продолжительность последнего контакта, сек.")
plt.ylabel("Вероятность / класс y")
plt.title("Однофакторная логистическая регрессия")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("02_logistic_curve.png", dpi=200)
plt.show()
 
# 13. Прогноз для нескольких новых объектов
# создаём набор гипотетических объектов с разной продолжительностью звонка
new_objects = pd.DataFrame({
    "duration": [60, 150, 300, 600, 1000]
})
# получаем предсказанный класс и вероятность класса "yes" для каждого объекта
new_pred = model.predict(new_objects)
new_prob = model.predict_proba(new_objects)[:, 1]
# собираем результаты в удобную для чтения таблицу
new_result = new_objects.copy()
new_result["Прогноз класса"] = np.where(new_pred == 1, "yes", "no")
new_result["Вероятность yes"] = new_prob
print("\nПрогноз для новых объектов:")
print(new_result.to_string(index=False))
plt.figure(figsize=(10, 6))
# точки-прогнозы, соединённые линией — как вероятность меняется с ростом duration
plt.plot(new_objects["duration"], new_prob, marker="o")
# подписываем каждую точку: предсказанный класс и его вероятность
for x, p, pred in zip(new_objects["duration"], new_prob, new_pred):
    plt.annotate(
        f"{'yes' if pred == 1 else 'no'}\n{p:.2f}",
        (x, p),
        textcoords="offset points",
        xytext=(0, 10),  # смещаем подпись немного вверх, чтобы не перекрывала точку
        ha="center"
    )
plt.xlabel("Продолжительность контакта, сек.")
plt.ylabel("Вероятность y=yes")
plt.title("Прогноз для новых объектов")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("03_new_predictions_single.png", dpi=200)
plt.show()

# 14. Средняя абсолютная, квадратичная ошибка и качество
y_test_pred = model.predict(X_test)
# абсолютная оценка ошибка
mae = mean_absolute_error(y_test, y_test_pred)
# квадратичная оценка ошибка
mse = mean_squared_error(y_test, y_test_pred)
accuracy = accuracy_score(y_test, y_test_pred)

print("\nОценка модели на тестовой выборке:")
print(f"MAE = {mae:.6f}")
print(f"MSE = {mse:.6f}")
print(f"Accuracy (качество) = {accuracy:.6f} ({accuracy*100:.2f}%)")

