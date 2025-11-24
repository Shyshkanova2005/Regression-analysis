import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#1. Зчитування даних 
df = pd.read_excel('flats_1.xlsx')
print(df)

#5. Створення квадратичних та кубічних ознак 
df['Площа ^2'] = df['Площа (м²)'] ** 2
df['Площа ^3'] = df['Площа (м²)'] ** 3

#2. Розбиття на тренувальну та валідаційну
train, test = train_test_split(df, test_size=0.3, random_state=42)

print("\nТренувальна вибірка (70%)")
print(train)
print("\nВалідаційна вибірка (30%)")
print(test)

#3. Навчання лінійної регресії та оцінка на валідаційній
linear_predictors = ['Площа (м²)']
outcome = ['Ціна ($)']

linear_model = LinearRegression()
linear_model.fit(train[linear_predictors], train[outcome])

print("\nЛінійна регресія (тренувальна вибірка)")
print(f"Константа: {linear_model.intercept_[0]:.2f}")
print(f"Коефіцієнт для {linear_predictors[0]}: {linear_model.coef_[0][0]:.2f}")

# Прогноз на валідаційній
y_pred_test = linear_model.predict(test[linear_predictors])

print("\nПрогнози на валідаційній вибірці:")
for real, pred in zip(test[outcome].values.flatten(), y_pred_test.flatten()):
    print(f"Реальна ціна: {real} -> Прогнозована ціна: {pred:.2f}")

mse_val = mean_squared_error(test[outcome], y_pred_test)
print(f"\nСередньоквадратична помилка: {mse_val:.2f}")

# 4 та 6. Оцінка моделей та крос-валідація
models = {
    'Лінійна': ['Площа (м²)'],
    'Квадратична': ['Площа (м²)', 'Площа ^2'],
    'Кубічна': ['Площа (м²)', 'Площа ^2', 'Площа ^3']
}

results_table = pd.DataFrame(columns=['Модель', 'Валідація', 'Крос-Валідація'])

for name, predictors in models.items():
    model = LinearRegression()
    model.fit(train[predictors], train[outcome])
    
    # Валідація
    y_pred_val = model.predict(test[predictors])
    mse_val = mean_squared_error(test[outcome], y_pred_val)
    
    # Крос-валідація
    cross = -cross_val_score(model, df[predictors], df[outcome], cv=3, scoring='neg_mean_squared_error')
    
    results_table.loc[len(results_table)] = [name, mse_val, cross.mean()]
    
    print(f"\n{name} регресія:")
    print(f"Середньоквадратична помилка на валідаційній: {mse_val:.2f}")
    print(f"Середньоквадратична помилка крос-валідації (3 блоки): {cross}, середнє: {cross.mean():.2f}")
    print("Коефіцієнти:", model.intercept_, model.coef_)

print("\nРезультати оцінки моделей:")
print(results_table)

#Вибір найкращої моделі
best_model_name = results_table.loc[results_table['Крос-Валідація'].idxmin(), 'Модель']
best_predictors = models[best_model_name]
print(f"\nНайкраща модель за крос-валідацією: {best_model_name}")

#7. Формули регресій
print("\nФормули регресій:")
for name, predictors in models.items():
    model = LinearRegression()
    model.fit(train[predictors], train[outcome])
    formula = f"{name} регресія: Ціна = {model.intercept_[0]:.2f}"
    for n, c in zip(predictors, model.coef_[0]):
        formula += f" + ({c:.2f}*{n})"
    print(formula)

#8.Графік
g = np.linspace(df['Площа (м²)'].min(), df['Площа (м²)'].max(), 300)
plt.figure(figsize=(9,6))
plt.scatter(df['Площа (м²)'], df['Ціна ($)'], color='black', label='Дані')

model_1 = LinearRegression().fit(train[['Площа (м²)']], train[outcome])
model_2 = LinearRegression().fit(train[['Площа (м²)', 'Площа ^2']], train[outcome])
model_3 = LinearRegression().fit(train[['Площа (м²)', 'Площа ^2', 'Площа ^3']], train[outcome])

plt.plot(g, model_1.predict(pd.DataFrame({'Площа (м²)': g})), label='Лінійна')
plt.plot(g, model_2.predict(pd.DataFrame({'Площа (м²)': g, 'Площа ^2': g**2})), label='Квадратична')
plt.plot(g, model_3.predict(pd.DataFrame({'Площа (м²)': g, 'Площа ^2': g**2, 'Площа ^3': g**3})), label='Кубічна')

plt.xlabel('Площа (м²)')
plt.ylabel('Ціна ($)')
plt.title('Площа і вартість квартир')
plt.legend()
plt.grid(True)
plt.show()

# 9. Краща модель для тестової вибірки
test_areas = pd.DataFrame({'Площа (м²)': [30, 50, 100]})
test_areas['Площа ^2'] = test_areas['Площа (м²)'] ** 2
test_areas['Площа ^3'] = test_areas['Площа (м²)'] ** 3

best_reg = LinearRegression().fit(df[best_predictors], df[outcome])
predicted_prices = best_reg.predict(test_areas[best_predictors])

print("\nПрогноз вартості для площ 30, 50, 100 кв.м (найкраща модель):")
for area, price in zip(test_areas['Площа (м²)'], predicted_prices):
    print(f"Площа {area} м² → Ціна: {price[0]:.2f} $")
