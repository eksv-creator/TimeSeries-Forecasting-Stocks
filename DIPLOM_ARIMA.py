# --- Часть 1: Эконометрический анализ ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# Загрузка данных
df = pd.read_csv("stocks.csv", sep=";", encoding='utf-8')
df['Data'] = pd.to_datetime(df['Data'], format='%d.%m.%Y')

# Очистка числовых столбцов
numeric_cols = df.columns.drop('Data')
for col in numeric_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(' ', '')
        .str.replace(',', '.')
        .astype(float)
    )

# Визуализация временных рядов
for col in numeric_cols:
    plt.figure(figsize=(10, 4))
    plt.plot(df['Data'], df[col], linewidth=1.2)
    plt.title(f' {col}', fontsize=14)
    plt.xlabel('Дата')
    plt.ylabel('Значение')
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

# Проверка нормальности
print("\n=== Тест Жарка–Бера на нормальность ===")
for col in numeric_cols:
    jb_stat, jb_p = stats.jarque_bera(df[col].dropna())
    norm = "не нормальное" if jb_p < 0.05 else "нормальное"
    print(f"• {col}: p = {jb_p:.4f} → {norm}")

# Стационарность: лог-приросты
df_indexed = df.set_index('Data')
df_log = np.log(df_indexed[numeric_cols])
df_stationary = df_log.diff().dropna()

print("\nДанные преобразованы в лог-доходность (лог-приросты).")

# Тест Дикки–Фуллера (ADF)
print("\n=== Тест Дикки–Фуллера на стационарность ===")
is_stationary_all = True
for col in df_stationary.columns:
    adf_result = adfuller(df_stationary[col].dropna(), autolag='AIC')
    p_value = adf_result[1]
    is_stationary = p_value < 0.05
    status = "стационарен" if is_stationary else "НЕ стационарен"
    print(f"• {col}: p = {p_value:.6f} → {status}")
    if not is_stationary:
        is_stationary_all = False

if is_stationary_all:
    print("\n✅ Все ряды стационарны.")
else:
    print("\n⚠ Некоторые ряды нестационарны!")

# Визуализация стационарных рядов
for col in df_stationary.columns:
    plt.figure(figsize=(10, 4))
    plt.plot(df_stationary.index, df_stationary[col], color='purple')
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f' {col}')
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()

# ACF / PACF для целевой переменной
y = df_stationary['Share price'].dropna()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(y, lags=20, ax=ax1)
plot_pacf(y, lags=20, ax=ax2)
plt.suptitle('ACF и PACF для лог-доходности акций')
plt.tight_layout()
plt.show()

# Целевая переменная
y = df_stationary['Share price'].copy()
X_other = df_stationary.drop(columns=['Share price'])

# Разделение (80/20)
split_idx = int(len(y) * 0.8)
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Подбор ARIMA
print("\n=== Ручной подбор ARIMA ===")
adf_result = adfuller(y_train)
print(f"ADF p-value = {adf_result[1]:.6f} → d = 0")

candidates = [(1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 0), (0, 0, 2), (2, 0, 1), (1, 0, 2)]
results = []
for order in candidates:
    try:
        model = ARIMA(y_train, order=order)
        fitted = model.fit()
        results.append((order, fitted.aic, fitted.bic))
    except:
        pass

if results:
    best_order = min(results, key=lambda x: x[1])[0]
    print(f"🏆 Лучшая модель: ARIMA{best_order} (по AIC)")

    final_arima = ARIMA(y_train, order=best_order).fit()

    # Прогноз на тест
    forecast = final_arima.get_forecast(steps=len(y_test))
    y_pred_arima = forecast.predicted_mean
    r2_arima = r2_score(y_test, y_pred_arima)
    mse_arima = mean_squared_error(y_test, y_pred_arima)
    print(f"R² (тест): {r2_arima:.4f}, MSE: {mse_arima:.6f}")

    # Вывод уравнения ARIMA
    params = final_arima.params
    const = params.get('const', 0)
    ar_terms = [params[f'ar.L{i + 1}'] if f'ar.L{i + 1}' in params else 0 for i in range(best_order[0])]
    ma_terms = [params[f'ma.L{i + 1}'] if f'ma.L{i + 1}' in params else 0 for i in range(best_order[2])]

    eq = f"y_t = {const:.4f}"
    for i, phi in enumerate(ar_terms):
        sign = " + " if phi >= 0 else " - "
        eq += f"{sign}{abs(phi):.4f} * y_{{t-{i + 1}}}"
    for i, theta in enumerate(ma_terms):
        sign = " + " if theta >= 0 else " - "
        eq += f"{sign}{abs(theta):.4f} * ε_{{t-{i + 1}}}"
    eq += " + ε_t"
    print("\nУравнение ARIMA:")
    print(eq)

    # График прогноза
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.index, y_test, label='Факт', color='blue')
    plt.plot(y_test.index, y_pred_arima, label=f'ARIMA{best_order}', linestyle='--', color='red')
    plt.title('Прогноз ARIMA на тестовой выборке')
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()

    # Прогноз на будущее
    steps_ahead = 20
    forecast_future = final_arima.get_forecast(steps=steps_ahead)
    forecast_mean = forecast_future.predicted_mean
    forecast_ci = forecast_future.conf_int()
    last_date = y.index[-1]
    forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=5), periods=steps_ahead, freq='5D')

    plt.figure(figsize=(12, 6))
    plt.plot(y.index, y, label='История', color='blue')
    plt.plot(forecast_index, forecast_mean, label='Прогноз ARIMA', color='red', linestyle='--')
    plt.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='red', alpha=0.2,
                     label='95% ДИ')
    plt.title(f'Прогноз ARIMA{best_order} на {steps_ahead} периодов вперёд')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("Не удалось подобрать ARIMA.")


# В конце Части 1, где у вас уже есть y_test и y_pred_arima
import pandas as pd

# Создаём DataFrame с датами и прогнозом ARIMA
arima_forecast_test = pd.DataFrame({
    'Date': y_test.index,         # убедитесь, что y_test — ваш тест из Части 1
    'y_pred_arima': y_pred_arima
}).set_index('Date')

# Сохраняем
arima_forecast_test.to_csv("arima_test_forecast.csv")
print("Прогноз ARIMA на тест сохранён в 'arima_test_forecast.csv'")