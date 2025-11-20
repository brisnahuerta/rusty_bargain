# 📋 Mejoras Sugeridas para Rusty Bargain

Este documento contiene recomendaciones para mejorar el proyecto de predicción de precios de vehículos.

---

## 🔴 PROBLEMAS CRÍTICOS

### 1. Imports innecesarios
**Ubicación:** Línea 108, 103

**Problema:**
- `f1_score` y `confusion_matrix` son para clasificación, no regresión
- `preprocessing` genérico nunca se usa

**Acción:**
```python
# Eliminar estas líneas:
from sklearn.metrics import mean_squared_error, r2_score, f1_score, confusion_matrix
from sklearn import preprocessing

# Reemplazar por:
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
```

**Prioridad:** ⭐⭐⭐ ALTA

---

### 2. Error en el modelo LightGBM
**Ubicación:** Línea 576

**Problema:**
El tiempo de entrenamiento usa `t0` del modelo Random Forest, no de LightGBM.

**Acción:**
```python
# Agregar ANTES de entrenar LightGBM (línea 557):
t0 = time.time()
model_lgbm = LGBMRegressor(...)
```

**Prioridad:** ⭐⭐⭐ ALTA

---

### 3. Inconsistencia en datos de entrenamiento
**Ubicación:** Líneas 520-578

**Problema:**
- Linear Regression y Random Forest usan `X_train_lr` (train + valid)
- LightGBM usa solo `train_chars` y `valid_chars` separados
- Las comparaciones no son justas

**Acción:**
Unificar el enfoque para todos los modelos o documentar claramente por qué son diferentes.

**Prioridad:** ⭐⭐⭐ ALTA

---

## 🟡 MEJORAS IMPORTANTES

### 4. Falta de documentación README
**Problema:**
No existe un archivo `README.md` que explique el proyecto.

**Acción:**
Crear `README.md` con:
- Descripción del proyecto
- Instalación de dependencias
- Cómo ejecutar el notebook
- Resultados principales
- Estructura del proyecto

**Prioridad:** ⭐⭐⭐ ALTA

---

### 5. Falta de .gitignore
**Problema:**
Archivos innecesarios podrían subirse a Git.

**Acción:**
Crear `.gitignore` con:
```
.venv/
.DS_Store
__pycache__/
*.pyc
.ipynb_checkpoints/
*.pkl
*.joblib
```

**Prioridad:** ⭐⭐⭐ ALTA

---

### 6. Gestión de outliers mejorable
**Ubicación:** Líneas 298-315, 361-381

**Problema:**
- Código repetitivo
- Línea 362: Usa `car_data['power']` en lugar de `data_model['power']`

**Acción:**
Crear función reutilizable:
```python
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

# Uso:
data_model_filtered = remove_outliers(car_data_filtered, 'price')
data_model_filtered = remove_outliers(data_model_filtered, 'power')
```

**Prioridad:** ⭐⭐ MEDIA

---

### 7. Feature Engineering limitado
**Ubicación:** Línea 324

**Problema:**
Eliminas columnas de fecha sin extraer información útil.

**Acción:**
Antes de eliminar, extraer features:
```python
# Convertir a datetime
car_data['date_created'] = pd.to_datetime(car_data['date_created'])
car_data['date_crawled'] = pd.to_datetime(car_data['date_crawled'])

# Crear nuevas features
car_data['vehicle_age'] = 2016 - car_data['registration_year']
car_data['days_since_posted'] = (car_data['date_crawled'] - car_data['date_created']).dt.days
```

**Prioridad:** ⭐⭐ MEDIA

---

### 8. Codificación de variables categóricas
**Ubicación:** Línea 432

**Problema:**
`OrdinalEncoder` para `model` y `brand` asume un orden que no existe.

**Acción:**
Usar Target Encoding o Frequency Encoding:
```python
# Target Encoding (requiere category_encoders)
from category_encoders import TargetEncoder
te = TargetEncoder(cols=['model', 'brand'])
data_encoded = te.fit_transform(data_model_filtered[['model', 'brand']], 
                                 data_model_filtered['price'])

# O Frequency Encoding (más simple):
for col in ['model', 'brand']:
    freq = data_model_filtered[col].value_counts(normalize=True)
    data_model_filtered[f'{col}_freq'] = data_model_filtered[col].map(freq)
```

**Prioridad:** ⭐⭐ MEDIA

---

### 9. Falta validación cruzada
**Problema:**
Solo un split único puede dar resultados sesgados.

**Acción:**
```python
from sklearn.model_selection import cross_val_score

# Para cada modelo:
cv_scores = cross_val_score(model, X, y, cv=5, 
                           scoring='neg_mean_squared_error')
rmse_cv = np.sqrt(-cv_scores.mean())
print(f"RMSE CV: {rmse_cv:.2f} (+/- {cv_scores.std():.2f})")
```

**Prioridad:** ⭐⭐ MEDIA

---

### 10. Métricas adicionales
**Problema:**
Solo usas RMSE.

**Acción:**
Agregar más métricas:
```python
from sklearn.metrics import mean_absolute_error, r2_score

def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape
    }
```

**Prioridad:** ⭐⭐ MEDIA

---

## 🟢 MEJORAS DE CÓDIGO Y ESTILO

### 11. Magic numbers
**Ubicación:** Líneas 335, 473

**Problema:**
Valores hardcodeados dificultan el mantenimiento.

**Acción:**
Definir constantes al inicio:
```python
# Configuración
MIN_YEAR = 1950
MAX_YEAR = 2025
TEST_SIZE = 0.2
RANDOM_STATE = 12345
```

**Prioridad:** ⭐ BAJA

---

### 12. Falta logging de resultados
**Problema:**
Resultados solo se imprimen, no se almacenan estructuradamente.

**Acción:**
```python
results = []
results.append({'Model': 'Linear Regression', **results_lr})
results.append({'Model': 'Random Forest', **results_rf})
results.append({'Model': 'LightGBM', **results_lgbm})

results_df = pd.DataFrame(results)
results_df.to_csv('model_results.csv', index=False)
print(results_df)
```

**Prioridad:** ⭐ BAJA

---

### 13. Escalado de features
**Problema:**
Importas `StandardScaler` pero nunca lo usas.

**Acción:**
Probar escalado para Linear Regression:
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_lr)
X_test_scaled = scaler.transform(X_test_lr)

model_lr.fit(X_train_scaled, y_train)
```

**Prioridad:** ⭐ BAJA

---

### 14. Guardado del modelo
**Problema:**
No guardas el mejor modelo.

**Acción:**
```python
import joblib

# Guardar el mejor modelo
joblib.dump(model_lgbm, 'best_model_lgbm.pkl')

# Cargar después:
# model = joblib.load('best_model_lgbm.pkl')
```

**Prioridad:** ⭐ BAJA

---

## 📈 MEJORAS DE ANÁLISIS

### 15. Visualizaciones adicionales
**Acción sugerida:**
```python
# Matriz de correlación
plt.figure(figsize=(12, 8))
sns.heatmap(data_model_filtered.corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.show()

# Importancia de features (LightGBM)
import matplotlib.pyplot as plt
feature_importance = pd.DataFrame({
    'feature': X_train_lr.columns,
    'importance': model_lgbm.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
plt.xlabel('Importancia')
plt.title('Top 10 Features Más Importantes')
plt.show()
```

**Prioridad:** ⭐ BAJA

---

### 16. Análisis de errores
**Acción sugerida:**
```python
# Análisis de residuos
residuals = y_test - test_preds_lgbm

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(test_preds_lgbm, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicciones')
plt.ylabel('Residuos')
plt.title('Gráfico de Residuos')

plt.subplot(1, 2, 2)
plt.hist(residuals, bins=50)
plt.xlabel('Residuos')
plt.title('Distribución de Residuos')
plt.show()

# Análisis por rango de precio
price_ranges = pd.cut(y_test, bins=5)
error_by_range = pd.DataFrame({
    'range': price_ranges,
    'error': np.abs(y_test - test_preds_lgbm)
}).groupby('range')['error'].mean()
print(error_by_range)
```

**Prioridad:** ⭐ BAJA

---

### 17. Velocidad de predicción
**Acción sugerida:**
```python
import time

# Medir tiempo de predicción
n_predictions = 1000
t0 = time.time()
for _ in range(n_predictions):
    _ = model_lgbm.predict(X_test_lr[:1])
prediction_time = (time.time() - t0) / n_predictions

print(f"Tiempo promedio de predicción: {prediction_time*1000:.2f}ms")
```

**Prioridad:** ⭐ BAJA

---

## ✅ LO QUE ESTÁ BIEN

- ✅ Buena estructura del notebook con markdown explicativo
- ✅ Análisis exploratorio detallado
- ✅ Comparación de múltiples modelos
- ✅ Uso de GridSearchCV para Random Forest
- ✅ Early stopping en LightGBM
- ✅ Conclusiones claras al final

---

## 📊 RESUMEN DE PRIORIDADES

### ⭐⭐⭐ ALTA (Hacer primero)
1. Corregir bug del `t0` en LightGBM
2. Eliminar imports innecesarios
3. Unificar datos de entrenamiento
4. Crear `README.md`
5. Crear `.gitignore`

### ⭐⭐ MEDIA (Hacer después)
6. Refactorizar eliminación de outliers
7. Agregar feature engineering de fechas
8. Mejorar encoding de variables categóricas
9. Implementar validación cruzada
10. Agregar más métricas

### ⭐ BAJA (Opcional)
11. Definir constantes para magic numbers
12. Crear DataFrame de resultados
13. Probar escalado de features
14. Guardar modelo final
15. Agregar visualizaciones adicionales
16. Análisis de errores
17. Medir velocidad de predicción

---

**Fecha de creación:** 2025-11-19
**Versión:** 1.0
