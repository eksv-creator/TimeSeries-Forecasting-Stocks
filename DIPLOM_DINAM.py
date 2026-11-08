# --- Часть 2: Динамический регрессионный анализ (F-тест вместо χ²) ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import f_regression  # <<< F-тест
import warnings

warnings.filterwarnings('ignore')

# Загрузка и предварительная обработка
df = pd.read_csv("stocks.csv", sep=";", encoding='utf-8')
df['Data'] = pd.to_datetime(df['Data'], format='%d.%m.%Y')
numeric_cols = df.columns.drop('Data')
for col in numeric_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(' ', '')
        .str.replace(',', '.')
        .astype(float)
    )

# Стационарные лог-приросты
df_indexed = df.set_index('Data')
df_log = np.log(df_indexed[numeric_cols])
df_stationary = df_log.diff().dropna()

# Генерация признаков
y = df_stationary['Share price'].copy()
X = df_stationary.drop(columns=['Share price']).copy()

X['lag1'] = y.shift(1)
X['lag2'] = y.shift(2)
X['ma5'] = y.rolling(5).mean()
X['ma10'] = y.rolling(10).mean()
X['vol5'] = y.rolling(5).std()
X['vol10'] = y.rolling(10).std()

data_full = pd.concat([y, X], axis=1).dropna()
y_clean = data_full['Share price']
X_clean = data_full.drop(columns=['Share price'])

# === 1. F-тест: оценка линейной значимости признаков ===
print("\n=== F-тест на линейную зависимость (ANOVA F-test) ===")
F_scores, p_values = f_regression(X_clean, y_clean)

f_test_df = pd.DataFrame({
    'Признак': X_clean.columns,
    'F-статистика': F_scores,
    'p-value': p_values
}).sort_values('F-статистика', ascending=False)

# Добавим пометку о статистической значимости
f_test_df['Значим (p<0.05)'] = f_test_df['p-value'] < 0.05

print(f_test_df.to_string(index=False, float_format="%.4f"))

# === 2. VIF: диагностика мультиколлинеарности ===
print("\n=== VIF (мультиколлинеарность) ===")
vif_data = pd.DataFrame()
vif_data["Признак"] = X_clean.columns
vif_data["VIF"] = [variance_inflation_factor(X_clean.values, i) for i in range(X_clean.shape[1])]
vif_data = vif_data.sort_values("VIF", ascending=False)
print(vif_data.to_string(index=False, float_format="%.2f"))

# === 3. Комбинированный отбор признаков ===
# Шаг 1: Удаляем признаки с VIF > 10
X_filtered = X_clean.copy()
while True:
    vifs = [variance_inflation_factor(X_filtered.values, i) for i in range(X_filtered.shape[1])]
    if max(vifs) <= 10:
        break
    drop_col = X_filtered.columns[np.argmax(vifs)]
    print(f"• Удалён признак '{drop_col}' из-за высокого VIF (>10)")
    X_filtered = X_filtered.drop(columns=[drop_col])

# Шаг 2: Среди оставшихся — берём только статистически значимые по F-тесту (p < 0.05)
f_test_filtered = f_test_df[f_test_df['Признак'].isin(X_filtered.columns)]
significant_features = f_test_filtered[f_test_filtered['Значим (p<0.05)']]['Признак']

# Если значимых признаков < 2, берём топ-5 по F-статистике (на всякий случай)
if len(significant_features) < 2:
    print("\n⚠️ Мало статистически значимых признаков. Берём топ-5 по F-статистике.")
    significant_features = f_test_filtered.head(5)['Признак']

X_final = X_filtered[significant_features]
y_final = y_clean.loc[X_final.index]

print(f"\n✅ Финальный набор признаков ({len(significant_features)} шт.):")
for feat in significant_features:
    f_val = f_test_df.loc[f_test_df['Признак'] == feat, 'F-статистика'].iloc[0]
    p_val = f_test_df.loc[f_test_df['Признак'] == feat, 'p-value'].iloc[0]
    print(f"  • {feat} (F = {f_val:.2f}, p = {p_val:.4f})")

# === 4. Разделение на обучение и тест (временной ряд!) ===
split_idx = int(len(y_final) * 0.8)
X_train, X_test = X_final.iloc[:split_idx], X_final.iloc[split_idx:]
y_train, y_test = y_final.iloc[:split_idx], y_final.iloc[split_idx:]

# === 5. Модели ===
# OLS
X_ols_train = sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_ols_train).fit()
X_ols_test = sm.add_constant(X_test)
y_pred_ols = ols_model.predict(X_ols_test)
r2_ols = r2_score(y_test, y_pred_ols)

# Ridge
ridge_cv = RidgeCV(alphas=np.logspace(-4, 4, 50), cv=5)
ridge_cv.fit(X_train, y_train)
ridge = Ridge(alpha=ridge_cv.alpha_)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
r2_ridge = r2_score(y_test, y_pred_ridge)

# Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=7, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)

# Важность признаков (RF)
rf_importance = pd.DataFrame({
    'Признак': X_train.columns,
    'Важность (RF)': rf.feature_importances_
}).sort_values('Важность (RF)', ascending=False)

print("\n=== Важность признаков (Random Forest) ===")
print(rf_importance.to_string(index=False, float_format="%.4f"))


# === 6. Диагностика моделей ===
def plot_diagnostics(y_true, y_pred, model_name, r2):
    residuals = y_true - y_pred
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, y_true, alpha=0.7, color='darkblue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title(f'{model_name}: Факт vs Прогноз\nR² = {r2:.3f}')
    plt.xlabel('Прогноз')
    plt.ylabel('Факт')
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f'{model_name}: Q-Q plot остатков')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


plot_diagnostics(y_test, y_pred_ols, "OLS", r2_ols)
plot_diagnostics(y_test, y_pred_ridge, "Ridge", r2_ridge)
plot_diagnostics(y_test, y_pred_rf, "Random Forest", r2_rf)

# === 7. Сравнение моделей ===
comparison = pd.DataFrame({
    'Модель': ['OLS', 'Ridge', 'Random Forest'],
    'R² (тест)': [r2_ols, r2_ridge, r2_rf],
    'MSE (тест)': [
        mean_squared_error(y_test, y_pred_ols),
        mean_squared_error(y_test, y_pred_ridge),
        mean_squared_error(y_test, y_pred_rf)
    ],
    'MAE (тест)': [
        mean_absolute_error(y_test, y_pred_ols),
        mean_absolute_error(y_test, y_pred_ridge),
        mean_absolute_error(y_test, y_pred_rf)
    ]
}).sort_values('R² (тест)', ascending=False)

print("\n=== Итоговое сравнение моделей на тестовой выборке ===")
print(comparison.to_string(index=False, float_format="%.4f"))

# === 8. МНОГОШАГОВЫЙ ПРОГНОЗ RANDOM FOREST НА БУДУЩЕЕ ===
print("\n=== Прогноз Random Forest на будущее (с доверительным интервалом) ===")

steps_ahead = 20  # Горизонт прогноза (в периодах)
last_date = y_final.index[-1]

# Берём последнюю строку данных (все признаки)
current_features = X_final.iloc[-1].copy()

# Хранилища
forecast_rf = []
lower_bounds = []
upper_bounds = []

# Рекурсивный прогноз
for step in range(steps_ahead):
    # Подготовка вектора признаков
    X_step = current_features.values.reshape(1, -1)

    # Предсказания отдельных деревьев → для ДИ
    tree_preds = np.array([tree.predict(X_step)[0] for tree in rf.estimators_])
    pred_mean = np.mean(tree_preds)
    pred_lower = np.percentile(tree_preds, 2.5)  # 95% ДИ
    pred_upper = np.percentile(tree_preds, 97.5)

    forecast_rf.append(pred_mean)
    lower_bounds.append(pred_lower)
    upper_bounds.append(pred_upper)

    # --- ОБНОВЛЕНИЕ ПРИЗНАКОВ ДЛЯ СЛЕДУЮЩЕГО ШАГА ---
    # Обновляем лаги
    if 'lag2' in current_features.index:
        current_features['lag2'] = current_features['lag1']
    if 'lag1' in current_features.index:
        current_features['lag1'] = pred_mean

    # Обновляем скользящие средние
    for window in [5, 10]:
        ma_col = f'ma{window}'
        if ma_col in current_features.index:
            # Берём последние (window-1) прогнозов + текущий
            recent = forecast_rf[-(window - 1):] if len(forecast_rf) >= (window - 1) else forecast_rf
            # Дополняем нулями, если не хватает истории
            padded = [0.0] * (window - 1 - len(recent)) + recent
            current_features[ma_col] = np.mean(padded + [pred_mean])

    # Обновляем волатильность
    for window in [5, 10]:
        vol_col = f'vol{window}'
        if vol_col in current_features.index:
            recent = forecast_rf[-window:] if len(forecast_rf) >= window else forecast_rf
            padded = [0.0] * (window - len(recent)) + recent
            current_features[vol_col] = np.std(padded + [pred_mean])

# Создаём индекс дат (предполагается шаг в 5 дней, как у вас)
forecast_index = pd.date_range(
    start=last_date + pd.Timedelta(days=5),
    periods=steps_ahead,
    freq='5D'
)

# Преобразуем в Series
forecast_series = pd.Series(forecast_rf, index=forecast_index)
lower_series = pd.Series(lower_bounds, index=forecast_index)
upper_series = pd.Series(upper_bounds, index=forecast_index)

# === ГРАФИК ===
plt.figure(figsize=(12, 6))

# Исторические данные (финальный y_final)
plt.plot(y_final.index, y_final, label='Исторические данные', color='blue', linewidth=1.5)

# Прогноз
plt.plot(forecast_series.index, forecast_series,
         label='Прогноз Random Forest', color='green', linestyle='--', linewidth=2)

# Доверительный интервал
plt.fill_between(
    forecast_series.index,
    lower_series,
    upper_series,
    color='green', alpha=0.2, label='95% доверительный интервал'
)

plt.title(f'Прогноз лог-доходности на {steps_ahead} периодов вперёд\n(Random Forest)')
plt.xlabel('Дата')
plt.ylabel('Лог-доходность')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# === Таблица прогноза ===
forecast_table = pd.DataFrame({
    'Дата': forecast_index,
    'Прогноз (лог-доходность)': forecast_rf,
    'Нижняя граница (2.5%)': lower_bounds,
    'Верхняя граница (97.5%)': upper_bounds
})
print("\nПрогноз Random Forest на будущее:")
print(forecast_table.to_string(index=False, float_format="%.5f"))

# --- Превращаем прогнозы sklearn в pandas.Series ---
y_pred_ridge = pd.Series(y_pred_ridge, index=y_test.index, name='Ridge')
y_pred_rf = pd.Series(y_pred_rf, index=y_test.index, name='RF')
y_pred_ols = pd.Series(y_pred_ols, index=y_test.index, name='OLS')  # добавлено

# --- Загружаем прогноз ARIMA из Части 1 ---
try:
    arima_pred = pd.read_csv("arima_test_forecast.csv", index_col='Date', parse_dates=True)
except FileNotFoundError:
    raise FileNotFoundError("Файл 'arima_test_forecast.csv' не найден. Запустите сначала Часть 1.")

# --- Берём ДАТЫ из ARIMA-теста как эталон ---
arima_test_dates = arima_pred.index

# --- Находим пересечение с текущим y_test ---
common_dates = arima_test_dates.intersection(y_test.index)

if len(common_dates) == 0:
    raise ValueError("Нет общих дат! Убедитесь, что обе части используют одни и те же исходные данные.")

# --- Фильтруем ВСЁ по этим датам ---
y_test_common = y_test.loc[common_dates]
y_pred_ols_common = y_pred_ols.loc[common_dates]
y_pred_ridge_common = y_pred_ridge.loc[common_dates]
y_pred_rf_common = y_pred_rf.loc[common_dates]
y_pred_arima_common = arima_pred.loc[common_dates, 'y_pred_arima']

# --- График сравнения ---
plt.figure(figsize=(14, 6))
plt.plot(y_test_common.index, y_test_common, label='Факт', color='blue', linewidth=2)
plt.plot(y_test_common.index, y_pred_ols_common, label='OLS', linestyle='--', color='blue')
plt.plot(y_test_common.index, y_pred_ridge_common, label='Ridge', linestyle='--', color='orange')
plt.plot(y_test_common.index, y_pred_rf_common, label='Random Forest', linestyle='--', color='green')
plt.plot(y_test_common.index, y_pred_arima_common, label='ARIMA', linestyle='--', color='red')

plt.title('Сравнение моделей на тестовой выборке', fontsize=14)
plt.xlabel('Дата')
plt.ylabel('Лог-доходность')
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# --- Таблица сравнения ---
from sklearn.metrics import r2_score
comparison_final = pd.DataFrame({
    'Модель': ['OLS', 'Ridge', 'Random Forest', 'ARIMA'],
    'R² (тест)': [
        r2_score(y_test_common, y_pred_ols_common),
        r2_score(y_test_common, y_pred_ridge_common),
        r2_score(y_test_common, y_pred_rf_common),
        r2_score(y_test_common, y_pred_arima_common)
    ]
}).sort_values('R² (тест)', ascending=False)

print("\n=== Окончательное сравнение моделей ===")
print(comparison_final.to_string(index=False, float_format="%.4f"))