# Rusty Bargain

Este repositorio contiene ejercicios de Data Science para estimar el valor de mercado de coches usados.

## Descripción

Rusty Bargain es un servicio de venta de coches de segunda mano que desarrolla una app para atraer a nuevos clientes. Con la app puedes averiguar rápidamente el valor de mercado de tu coche, con acceso al historial, especificaciones técnicas, versiones de equipamiento y precios.

La empresa se enfoca en:

- La calidad de la predicción
- La velocidad de la predicción
- El tiempo requerido para el entrenamiento

## Objetivo

Crear un modelo que determine el valor de mercado de los coches usados.

## Herramientas utilizadas

- Python y Jupyter Notebook para el análisis interactivo
- Pandas y NumPy para la manipulación de datos
- Scikit-learn para algoritmos de aprendizaje automático
- LightGBM para modelos de gradient boosting eficientes

El conjunto de datos principal es `car_data.csv`, con campos como `DateCrawled`, `Price`, `VehicleType`, `RegistrationYear`, `Power`, `Model` y `Mileage`.

## Modelos desarrollados

- **Regresión Lineal** como modelo base
- **Random Forest** con búsqueda de hiperparámetros
- **LightGBM** como modelo principal por su equilibrio entre calidad y velocidad

Durante las pruebas, la Regresión Lineal alcanzó un RMSE de ~2246, el Random Forest redujo el error a ~1338 y LightGBM logró el mejor equilibrio con un RMSE cercano a 1306 y un tiempo de entrenamiento de ~358 segundos.

## Uso

Ejecuta el cuaderno `rusty_bargain.ipynb` para reproducir el análisis y entrenar los modelos.
