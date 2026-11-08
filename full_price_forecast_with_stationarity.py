import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error
)
import warnings
warnings.filterwarnings('ignore')
#Блок 1
# Загрузка данных
df = pd.read_csv("stocks.csv", sep=";", encoding='utf-8')

# Преобразование даты
df['Data'] = pd.to_datetime(df['Data'], format='%d.%m.%Y')

# Очистка числовых столбцов: убираем пробелы и заменяем ',' на '.'
numeric_cols = df.columns.drop('Data')
for col in numeric_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(' ', '')
        .str.replace(',', '.')
        .astype(float)
    )

# 3. Визуализация временных рядов
for col in numeric_cols:
    plt.figure(figsize=(10, 4))
    plt.plot(df['Data'], df[col], linewidth=1.2)
    plt.title(f'Динамика переменной: {col}', fontsize=14)
    plt.xlabel('Дата')
    plt.ylabel('Значение')
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

# 4. Проверка нормальности распределения
print("\nТест на нормальность:")
jb_results = []
for col in numeric_cols:
    jb_stat, jb_p = stats.jarque_bera(df[col].dropna())
    jb_results.append({'Переменная': col, 'p-value': jb_p})
    norm = "не нормальное" if jb_p < 0.05 else "нормальное"
    print(f"• {col}: p-value = {jb_p:.4f} → распределение {norm}")

# Визуализация: гистограммы + Q-Q plot
print("\n=== Визуализация нормальности ===")
for col in numeric_cols:
    data = df[col].dropna()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Гистограмма
    ax1.hist(data, bins=20, density=True, color='pink', alpha=0.7, edgecolor='black')
    mu, sigma = data.mean(), data.std()
    x = np.linspace(data.min(), data.max(), 100)
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r--', linewidth=2)
    ax1.set_title(f'Гистограмма: {col}')
    ax1.grid(True, alpha=0.3)

    # Q-Q plot
    stats.probplot(data, dist="norm", plot=ax2)
    ax2.set_title(f'Q-Q plot: {col}')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# 5. Корреляционная матрица
print("\nКорреляционная матрица:")
corr = df[numeric_cols].corr()

# Тепловая карта в розовом градиенте
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    cmap="RdPu",
    fmt=".2f",
    linewidths=0.5,
    linecolor="white",
    vmin=-1, vmax=1
)
plt.title("Корреляционная матрица", fontsize=14, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=300)
plt.show()

print(corr)

#БЛОК 2

# 6. Стационарность данных
# Устанавливаем дату как индекс
df_indexed = df.set_index('Data')

# Логарифмируем ВСЕ числовые столбцы (только положительные значения)
df_log = np.log(df_indexed[numeric_cols])

# Первая разность (лог-приросты)
df_stationary = df_log.diff()

# Применяем вторую разность для нестационарных рядов
for col in df_stationary.columns:
    adf_result = adfuller(df_stationary[col].dropna(), autolag='AIC')
    p_value = adf_result[1]

    if p_value >= 0.05:
        print(f"⚠ {col} не стационарен (p={p_value:.3f}) → применяем вторую разность")
        df_stationary[col] = df_log[col].diff().diff()

# Убираем NaN, возникшие после разностей
df_stationary = df_stationary.dropna()

print("\nДанные преобразованы в лог-приросты (вторая разность применена при необходимости)")
print("Первые 5 наблюдений стационарных данных:")
print(df_stationary.head())

# 7. Проверка стационарности: Тест Дикки-Фуллера (ADF)
print("\n=== Тест Дикки–Фуллера (ADF) на стационарность ===")
is_stationary_all = True
for col in df_stationary.columns:
    adf_result = adfuller(df_stationary[col].dropna(), autolag='AIC')
    p_value = adf_result[1]
    is_stationary = p_value < 0.05
    status = "стационарен" if is_stationary else "НЕ стационарен"
    print(f"• {col}: p-value = {p_value:.6f} → {status}")
    if not is_stationary:
        is_stationary_all = False

if is_stationary_all:
    print("\nВсе временные ряды стационарны — можно строить модели!")
else:
    print("\n⚠ Некоторые ряды всё ещё нестационарны. Рассмотрите другие методы.")

# 7a. Визуализация стационарных данных
for col in df_stationary.columns:
    plt.figure(figsize=(10, 4))
    plt.plot(df_stationary.index, df_stationary[col], linewidth=1.0, color='purple')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title(f'Первая разность: {col}', fontsize=14)
    plt.xlabel('Дата')
    plt.ylabel('Лог-доходность')
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

# 8. Создание доп. признаков (feature engineering)
# Целевая переменная: лог-доходность акций
y = df_stationary['Share price'].copy()

# Исходные макрофакторы (в лог-приростах)
X = df_stationary.drop(columns=['Share price']).copy()

# Лаги доходности (инерция рынка)
X['lag1'] = y.shift(1)
X['lag2'] = y.shift(2)

# Скользящие средние (оценка ожидаемой доходности)
X['ma5'] = y.rolling(window=5).mean()   # средняя доходность за 5 дней
X['ma10'] = y.rolling(window=10).mean() # за 10 дней

# Волатильность (риск, стандартное отклонение)
X['vol5'] = y.rolling(window=5).std()
X['vol10'] = y.rolling(window=10).std()

# Удаляем строки с NaN (появились из-за lag и rolling)
data_full = pd.concat([y, X], axis=1).dropna()
y_clean = data_full['Share price']
X_clean = data_full.drop(columns=['Share price'])

print(f"\nФинальный размер выборки: {X_clean.shape[0]} наблюдений, {X_clean.shape[1]} признаков.")

# 9. Корреляция на стационарных данных
corr_stationary = data_full.corr()

# Тепловая карта
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_stationary, dtype=bool))
sns.heatmap(
    corr_stationary,
    mask=mask,
    annot=True,
    cmap="RdPu",
    fmt=".2f",
    linewidths=0.5,
    linecolor="white",
    vmin=-1, vmax=1
)
plt.title("Корреляционная матрица (лог-приросты)", fontsize=14, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("correlation_stationary.png", dpi=300)
plt.show()

# Корреляция с целевой переменной
target_corr = corr_stationary['Share price'].drop('Share price').sort_values(key=abs, ascending=False)
print("\nКорреляция признаков с доходностью акций (по модулю):")
print(target_corr)

# 10. Диагностика мультиколлинеарности: VIF
print("\n=== Коэффициент инфляции дисперсии (VIF) ===")
# Подготовка данных для VIF (без константы)
X_vif = X_clean.copy()

# Вычисление VIF для каждого признака
vif_data = pd.DataFrame()
vif_data["Признак"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

# Сортировка по убыванию VIF
vif_data = vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)
print(vif_data)

# Интерпретация
high_vif = vif_data[vif_data["VIF"] > 5]
if not high_vif.empty:
    print(f"\nОбнаружена мультиколлинеарность (VIF > 5) в {len(high_vif)} признаках.")
    print("Рекомендуется удалить один из сильно коррелирующих признаков (например, с наименьшей корреляцией с y).")
else:
    print("\nМультиколлинеарность отсутствует (все VIF < 5).")

# 11. Предварительный отбор признаков
print("\n=== Предварительный отбор признаков ===")

# Шаг 1: Удаляем признаки с VIF > 10 (сильная мультиколлинеарность)
X_filtered = X_vif.copy()
while True:
    vif_values = [variance_inflation_factor(X_filtered.values, i) for i in range(X_filtered.shape[1])]
    max_vif = max(vif_values)
    if max_vif > 10:
        drop_col = X_filtered.columns[vif_values.index(max_vif)]
        print(f"• Удалён признак '{drop_col}' (VIF = {max_vif:.2f})")
        X_filtered = X_filtered.drop(columns=[drop_col])
    else:
        break

# Шаг 2: Оставляем топ-N признаков по корреляции с y (например, топ-5)
N_FEATURES = 5
top_features = target_corr.head(N_FEATURES).index
final_features = [col for col in top_features if col in X_filtered.columns]

# Финальный набор признаков
X_final = X_filtered[final_features]
y_final = y_clean.loc[X_final.index]

print(f"\nФинальный набор признаков ({len(final_features)} шт.):")
for i, feat in enumerate(final_features, 1):
    corr_val = target_corr[feat]
    print(f"  {i}. {feat} (корреляция с y: {corr_val:.3f})")

print(f"\nФинальный датасет готов для построения OLS, Ridge, Random Forest и ARIMA.")
print(f"Размер: {X_final.shape[0]} x {X_final.shape[1]}")

# 11b. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ (временной ряд!)
print("\n=== РАЗДЕЛЕНИЕ ДАННЫХ: 80% обучение, 20% тест ===")
split_idx = int(len(y_final) * 0.8)
X_train, X_test = X_final.iloc[:split_idx], X_final.iloc[split_idx:]
y_train, y_test = y_final.iloc[:split_idx], y_final.iloc[split_idx:]
print(f"Обучающая выборка: {len(y_train)} наблюдений")
print(f"Тестовая выборка: {len(y_test)} наблюдений")

# === АВТОМАТИЧЕСКИЙ ПОДБОР
from pmdarima import auto_arima

print("\n=== ПОДГОТОВКА ARIMAX ДЛЯ МНОГОШАГОВОГО ПРОГНОЗА ===")

# 1️⃣ Стационарность экзогенных признаков
X_arimax = X_final.copy()
for col in X_arimax.columns:
    X_arimax[col] = X_arimax[col].diff().dropna()  # первая разность для всех признаков

# Убираем NaN, появившиеся после разности
X_arimax = X_arimax.dropna()
y_arimax = y_clean.loc[X_arimax.index]

# 2️⃣ Разделение на обучение и тест
split_idx = int(len(y_arimax) * 0.8)
X_arimax_train, X_arimax_test = X_arimax.iloc[:split_idx], X_arimax.iloc[split_idx:]
y_arimax_train, y_arimax_test = y_arimax.iloc[:split_idx], y_arimax.iloc[split_idx:]

# 3️⃣ Подбор ARIMAX
arimax_model = auto_arima(
    y_arimax_train,
    exogenous=X_arimax_train,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    trace=True,
    max_p=3, max_q=3,
    d=0  # так как y уже стационарен
)

print(f"\n🏆 Лучшая ARIMAX модель:\n{arimax_model.summary()}")

# 4️⃣ Прогноз на тест
y_pred_arimax = arimax_model.predict(n_periods=len(y_arimax_test), exogenous=X_arimax_test)
r2_arimax_test = r2_score(y_arimax_test, y_pred_arimax)
mse_arimax_test = mean_squared_error(y_arimax_test, y_pred_arimax)
mae_arimax_test = mean_absolute_error(y_arimax_test, y_pred_arimax)

print(f"\nARIMAX - R² (тест): {r2_arimax_test:.4f}")
print(f"ARIMAX - MSE (тест): {mse_arimax_test:.6f}")
print(f"ARIMAX - MAE (тест): {mae_arimax_test:.6f}")

# 5️⃣ Многошаговый прогноз на будущее
steps_ahead = 20  # количество шагов вперед
# Берем последние значения признаков для рекурсивного прогноза
X_future = X_arimax_test.iloc[-1:].copy()

forecast_arimax = []
for step in range(steps_ahead):
    pred = arimax_model.predict(n_periods=1, exogenous=X_future)
    forecast_arimax.append(pred[0])

    # Рекурсивное обновление признаков (например, лаги)
    if 'lag1' in X_future.columns:
        X_future['lag2'] = X_future['lag1']
        X_future['lag1'] = pred[0]
    for ma in [5, 10]:
        ma_col = f'ma{ma}'
        if ma_col in X_future.columns:
            # обновляем скользящее среднее на основе предыдущих прогнозов
            history = forecast_arimax[-(ma-1):] if len(forecast_arimax) >= (ma-1) else forecast_arimax
            history_full = ([0.0]*(ma-1-len(history))) + history
            X_future[ma_col] = np.mean(history_full + [pred[0]])
    for vol in [5, 10]:
        vol_col = f'vol{vol}'
        if vol_col in X_future.columns:
            history = forecast_arimax[-vol:] if len(forecast_arimax) >= vol else forecast_arimax
            history_full = ([0.0]*(vol-len(history))) + history
            X_future[vol_col] = np.std(history_full)

# Индекс прогнозов
last_date = y_arimax.index[-1]
forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=5), periods=steps_ahead, freq='5D')
forecast_series = pd.Series(forecast_arimax, index=forecast_index)

# 6️⃣ Визуализация
plt.figure(figsize=(12, 6))
plt.plot(y_arimax.index, y_arimax, label='Исторические данные', color='blue')
plt.plot(forecast_series.index, forecast_series, label='Прогноз ARIMAX', color='red', linestyle='--')
plt.title(f'Многошаговый прогноз ARIMAX на {steps_ahead} периодов')
plt.xlabel('Дата')
plt.ylabel('Лог-доходность')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 12. МНОЖЕСТВЕННАЯ ЛИНЕЙНАЯ РЕГРЕССИЯ (на обучающей выборке)
print("\n=== МНОЖЕСТВЕННАЯ ЛИНЕЙНАЯ РЕГРЕССИЯ (OLS) ===")
X_ols_train = sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_ols_train).fit()

# Прогноз на тестовой выборке
X_ols_test = sm.add_constant(X_test)
y_pred_ols = ols_model.predict(X_ols_test)

# Метрики на тесте
r2_ols_test = r2_score(y_test, y_pred_ols)
mse_ols_test = mean_squared_error(y_test, y_pred_ols)

print(ols_model.summary())
print(f"\nOLS - R² (тест): {r2_ols_test:.4f}")
print(f"OLS - MSE (тест): {mse_ols_test:.6f}")

# Уравнение модели
params = ols_model.params
equation = f"y = {params['const']:.6f}"
for col in X_train.columns:
    sign = " + " if params[col] >= 0 else " - "
    equation += f"{sign}{abs(params[col]):.6f} * {col}"
print(f"\nУравнение OLS:")
print(equation)

# 13. RIDGE-РЕГРЕССИЯ (на обучающей выборке)
print("\n=== RIDGE-РЕГРЕССИЯ ===")
ridge_cv = RidgeCV(alphas=np.logspace(-4, 4, 50), cv=5)
ridge_cv.fit(X_train, y_train)

ridge_best = Ridge(alpha=ridge_cv.alpha_)
ridge_best.fit(X_train, y_train)

# Прогноз на тесте
y_pred_ridge = ridge_best.predict(X_test)
r2_ridge_test = r2_score(y_test, y_pred_ridge)
mse_ridge_test = mean_squared_error(y_test, y_pred_ridge)

print(f"Лучший alpha: {ridge_cv.alpha_:.6f}")
print(f"Ridge - R² (тест): {r2_ridge_test:.4f}")
print(f"Ridge - MSE (тест): {mse_ridge_test:.6f}")

# Уравнение Ridge
ridge_equation = f"y = {ridge_best.intercept_:.6f}"
for i, col in enumerate(X_train.columns):
    sign = " + " if ridge_best.coef_[i] >= 0 else " - "
    ridge_equation += f"{sign}{abs(ridge_best.coef_[i]):.6f} * {col}"
print(f"\nУравнение Ridge:")
print(ridge_equation)

# 14. РУЧНОЙ ПОДБОР ПАРАМЕТРОВ ARIMA (на обучающей выборке)
print("\n=== РУЧНОЙ ПОДБОР ПАРАМЕТРОВ ARIMA ===")
adf_result = adfuller(y_train)
print(f"ADF p-value для y_train: {adf_result[1]:.6f} → ряд стационарен → d = 0")

# Подбор на обучающей
candidates = [(1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 0), (0, 0, 2), (1,1,1)]
results = []
for order in candidates:
    try:
        model = ARIMA(y_train, order=order)
        fitted = model.fit()
        aic = fitted.aic
        bic = fitted.bic
        results.append((order, aic, bic))
    except:
        pass

if results:
    best_order = min(results, key=lambda x: x[1])[0]
    print(f"🏆 Лучшая модель по AIC: ARIMA{best_order}")

    # Обучение на всей обучающей
    final_arima = ARIMA(y_train, order=best_order).fit()
    # Вывод уравнения ARIMA
    print(f"\nУравнение ARIMA{best_order}:")
    if best_order[0] > 0 and best_order[2] > 0:
        phi = final_arima.params['ar.L1']
        theta = final_arima.params['ma.L1']
        const = final_arima.params.get('const', 0)
        print(
            f"y_t = {const:.4f} {'+' if phi >= 0 else '-'} {abs(phi):.4f}*y_{{t-1}} + ε_t {'+' if theta >= 0 else '-'} {abs(theta):.4f}*ε_{{t-1}}")
    elif best_order[0] > 0:
        phi = final_arima.params['ar.L1']
        const = final_arima.params.get('const', 0)
        print(f"y_t = {const:.4f} {'+' if phi >= 0 else '-'} {abs(phi):.4f}*y_{{t-1}} + ε_t")
    elif best_order[2] > 0:
        theta = final_arima.params['ma.L1']
        const = final_arima.params.get('const', 0)
        print(f"y_t = {const:.4f} + ε_t {'+' if theta >= 0 else '-'} {abs(theta):.4f}*ε_{{t-1}}")
    else:
        print("y_t = ε_t (белый шум)")
    # Прогноз на длину тестовой выборки
    forecast = final_arima.get_forecast(steps=len(y_test))
    y_pred_arima = forecast.predicted_mean
    r2_arima_test = r2_score(y_test, y_pred_arima)
    mse_arima_test = mean_squared_error(y_test, y_pred_arima)
else:
    print("Не удалось подобрать ARIMA")

# 15. RANDOM FOREST (на обучающей выборке)

# 15. RANDOM FOREST НА ВСЕХ СТАЦИОНАРНЫХ ПРИЗНАКАХ
print("\n=== RANDOM FOREST НА ВСЕХ СТАЦИОНАРНЫХ ПРИЗНАКАХ ===")

X_rf = data_full.drop(columns=['Share price'])
y_rf = data_full['Share price']

# Разделение на обучение и тест (80/20)
split_idx = int(len(y_rf) * 0.8)
X_train_rf, X_test_rf = X_rf.iloc[:split_idx], X_rf.iloc[split_idx:]
y_train_rf, y_test_rf = y_rf.iloc[:split_idx], y_rf.iloc[split_idx:]

# Обучение модели
rf_all = RandomForestRegressor(n_estimators=200, max_depth=7, random_state=42)
rf_all.fit(X_train_rf, y_train_rf)

# Прогноз на тесте
y_pred_rf_all = rf_all.predict(X_test_rf)
r2_rf_all = r2_score(y_test_rf, y_pred_rf_all)
mse_rf_all = mean_squared_error(y_test_rf, y_pred_rf_all)
mae_rf_all = mean_absolute_error(y_test_rf, y_pred_rf_all)

print(f"Random Forest (все признаки) - R² (тест): {r2_rf_all:.4f}")
print(f"Random Forest (все признаки) - MSE (тест): {mse_rf_all:.6f}")
print(f"Random Forest (все признаки) - MAE (тест): {mae_rf_all:.6f}")

# Важность признаков
feature_importance_all = pd.DataFrame({
    'Признак': X_train_rf.columns,
    'Важность': rf_all.feature_importances_
}).sort_values(by='Важность', ascending=False)
print("\nВажность признаков (Random Forest, все признаки):")
print(feature_importance_all)




# 16. ГРАФИКИ ДИАГНОСТИКИ НА ТЕСТОВОЙ ВЫБОРКЕ
print("\n=== ГРАФИКИ ДИАГНОСТИКИ НА ТЕСТОВОЙ ВЫБОРКЕ ===")

# === OLS диагностика (на тесте) ===
residuals_ols = y_test - y_pred_ols
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.scatter(y_pred_ols, y_test, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f'OLS: Факт vs Прогноз (тест)\nR² = {r2_ols_test:.3f}')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.scatter(y_pred_ols, residuals_ols, alpha=0.7, color='purple')
plt.axhline(0, color='red', linestyle='--')
plt.title('OLS: Остатки vs Прогноз (тест)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
stats.probplot(residuals_ols, dist="norm", plot=plt)
plt.title('OLS: Q-Q plot остатков (тест)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# === Ridge диагностика ===
residuals_ridge = y_test - y_pred_ridge
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_pred_ridge, y_test, alpha=0.7, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f'Ridge: Факт vs Прогноз (тест)\nR² = {r2_ridge_test:.3f}')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
stats.probplot(residuals_ridge, dist="norm", plot=plt)
plt.title('Ridge: Q-Q plot остатков (тест)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# === Random Forest диагностика ===
residuals_rf = y_test - y_pred_rf_all
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_pred_rf_all, y_test, alpha=0.7, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f'Random Forest: Факт vs Прогноз (тест)\n')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
stats.probplot(residuals_rf, dist="norm", plot=plt)
plt.title('Random Forest: Q-Q plot остатков (тест)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 17. ПРОГНОЗ ARIMA НА БУДУЩЕЕ (на основе всей обучающей выборки)
print("\n=== ПРОГНОЗ ARIMA НА БУДУЩЕЕ (на 10 периодов) ===")
steps_ahead = 20
forecast_future = final_arima.get_forecast(steps=steps_ahead)
forecast_mean = forecast_future.predicted_mean
forecast_ci = forecast_future.conf_int()

last_date = y_final.index[-1]
forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=5), periods=steps_ahead, freq='5D')

plt.figure(figsize=(12, 6))
plt.plot(y_final.index, y_final, label='Исторические данные', color='blue')
plt.plot(forecast_index, forecast_mean, label='Прогноз ARIMA', color='red', linestyle='--')
plt.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1],
                 color='red', alpha=0.2, label='95% ДИ')
plt.title(f'Прогноз доходности на {steps_ahead} периодов вперёд (ARIMA{best_order})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 20. МНОГОШАГОВЫЙ ПРОГНОЗ RANDOM FOREST (все признаки)
print("\n=== МНОГОШАГОВЫЙ ПРОГНОЗ RANDOM FOREST НА ВСЕХ ПРИЗНАКАХ ===")

steps_ahead = 20  # количество прогнозируемых шагов
last_date = y_rf.index[-1]

# Берём последние доступные значения признаков
current_row = data_full.loc[y_rf.index[-1]].copy()

# Инициализация лагов, если их нет
for lag in ['lag1', 'lag2']:
    if lag not in current_row.index:
        current_row[lag] = 0.0

# Хранилища прогнозов
forecast_rf_all = []
forecast_rf_lower_all = []
forecast_rf_upper_all = []

# Рекурсивный прогноз
for step in range(steps_ahead):
    X_step = current_row[X_rf.columns].values.reshape(1, -1)

    tree_preds = np.array([tree.predict(X_step)[0] for tree in rf_all.estimators_])

    pred_mean = np.mean(tree_preds)
    pred_lower = np.percentile(tree_preds, 2.5)
    pred_upper = np.percentile(tree_preds, 97.5)

    forecast_rf_all.append(pred_mean)
    forecast_rf_lower_all.append(pred_lower)
    forecast_rf_upper_all.append(pred_upper)

    # --- ОБНОВЛЕНИЕ ЛАГОВ ---
    if 'lag2' in X_rf.columns:
        current_row['lag2'] = current_row['lag1']
    if 'lag1' in X_rf.columns:
        current_row['lag1'] = pred_mean

    # --- ОБНОВЛЕНИЕ СКОЛЬЗЯЩИХ СРЕДНИХ ---
    for ma in [5, 10]:
        ma_col = f'ma{ma}'
        if ma_col in X_rf.columns:
            history = forecast_rf_all[-(ma-1):] if len(forecast_rf_all) >= (ma-1) else forecast_rf_all
            history_full = ([0.0]*(ma-1-len(history))) + history
            current_row[ma_col] = np.mean(history_full + [pred_mean])

    # --- ОБНОВЛЕНИЕ ВОЛАТИЛЬНОСТИ ---
    for vol in [5, 10]:
        vol_col = f'vol{vol}'
        if vol_col in X_rf.columns:
            history = forecast_rf_all[-vol:] if len(forecast_rf_all) >= vol else forecast_rf_all
            history_full = ([0.0]*(vol-len(history))) + history
            current_row[vol_col] = np.std(history_full)

# Индекс прогноза
forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=5),
                               periods=steps_ahead, freq='5D')

forecast_rf_series_all = pd.Series(forecast_rf_all, index=forecast_index)
forecast_rf_lower_series_all = pd.Series(forecast_rf_lower_all, index=forecast_index)
forecast_rf_upper_series_all = pd.Series(forecast_rf_upper_all, index=forecast_index)

# График прогноза
plt.figure(figsize=(12, 6))
plt.plot(y_rf.index, y_rf, label='Исторические данные', color='blue', linewidth=1.5)
plt.plot(forecast_rf_series_all.index, forecast_rf_series_all,
         label='Прогноз Random Forest (все признаки)', color='green', linestyle='--', linewidth=2)
plt.fill_between(
    forecast_rf_series_all.index,
    forecast_rf_lower_series_all,
    forecast_rf_upper_series_all,
    color='green', alpha=0.2, label='95% доверительный интервал'
)
plt.title(f'Прогноз доходности акций на {steps_ahead} периодов вперед\nRandom Forest (все признаки)')
plt.xlabel('Дата')
plt.ylabel('Лог-доходность')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Таблица прогноза
rf_forecast_df_all = pd.DataFrame({
    'Дата': forecast_index,
    'Прогноз (лог-доходность)': forecast_rf_all,
    'Нижняя граница (95%)': forecast_rf_lower_all,
    'Верхняя граница (95%)': forecast_rf_upper_all
})
print(f"\nПрогноз Random Forest на {steps_ahead} периодов (все признаки):")
print(rf_forecast_df_all.to_string(index=False))

# --------------------------------------------------
# СРАВНЕНИЕ МОДЕЛЕЙ НА ТЕСТОВОЙ ВЫБОРКЕ (с ARIMAX)
# --------------------------------------------------
print("\n=== СРАВНЕНИЕ МОДЕЛЕЙ НА ТЕСТОВОЙ ВЫБОРКЕ ===")

# OLS
X_ols_test = sm.add_constant(X_test)
y_pred_ols = ols_model.predict(X_ols_test)
r2_ols_test = r2_score(y_test, y_pred_ols)
mse_ols_test = mean_squared_error(y_test, y_pred_ols)
mae_ols_test = mean_absolute_error(y_test, y_pred_ols)

# Ridge
y_pred_ridge = ridge_best.predict(X_test)
r2_ridge_test = r2_score(y_test, y_pred_ridge)
mse_ridge_test = mean_squared_error(y_test, y_pred_ridge)
mae_ridge_test = mean_absolute_error(y_test, y_pred_ridge)

# Random Forest на всех признаках
y_pred_rf_all = rf_all.predict(X_test_rf)
r2_rf_all = r2_score(y_test_rf, y_pred_rf_all)
mse_rf_all = mean_squared_error(y_test_rf, y_pred_rf_all)
mae_rf_all = mean_absolute_error(y_test_rf, y_pred_rf_all)

# ARIMAX
y_pred_arimax = arimax_model.predict(n_periods=len(y_test), exogenous=X_arimax_test)
r2_arimax_test = r2_score(y_test, y_pred_arimax)
mse_arimax_test = mean_squared_error(y_test, y_pred_arimax)
mae_arimax_test = mean_absolute_error(y_test, y_pred_arimax)

# Объединяем все метрики в таблицу
comparison_df = pd.DataFrame({
    'Модель': ['OLS', 'Ridge', 'Random Forest (все признаки)', 'ARIMAX'],
    'R² (тест)': [r2_ols_test, r2_ridge_test, r2_rf_all, r2_arimax_test],
    'MSE (тест)': [mse_ols_test, mse_ridge_test, mse_rf_all, mse_arimax_test],
    'MAE (тест)': [mae_ols_test, mae_ridge_test, mae_rf_all, mae_arimax_test]
}).sort_values('R² (тест)', ascending=False)

print(comparison_df.to_string(index=False, float_format="%.4f"))

# --------------------------------------------------
# Графическое сравнение прогнозов
# --------------------------------------------------
plt.figure(figsize=(14, 6))
plt.plot(y_test.index, y_test, label='Факт', color='blue', linewidth=1.5)
plt.plot(y_test.index, y_pred_ols, label='OLS', linestyle='--')
plt.plot(y_test.index, y_pred_ridge, label='Ridge', linestyle='--')
plt.plot(y_test_rf.index, y_pred_rf_all, label='RF (все признаки)', linestyle='--')
plt.plot(y_test.index, y_pred_arimax, label='ARIMAX', linestyle='--')

plt.title('Сравнение моделей на тестовой выборке')
plt.xlabel('Дата')
plt.ylabel('Лог-доходность')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
