import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Загрузка модели
with open('realty_price_model_v2.pkl', 'rb') as f:
    model = pickle.load(f)

# Интерфейс Streamlit
st.title("Прогноз стоимости недвижимости")

# Поля для ввода признаков
total_square = st.number_input("Введите общую площадь (м²)", min_value=1.0, step=1.0)
rooms = st.number_input("Введите количество комнат", min_value=1, step=1)
floor = st.number_input("Введите этаж", min_value=1, step=1)
lat = st.number_input("Введите широту", format="%.6f")
lon = st.number_input("Введите долготу", format="%.6f")
city = st.text_input("Введите город")
district = st.text_input("Введите район")

# Кнопка для предсказания
if st.button("Предсказать"):
    # Формируем массив признаков и преобразуем в DataFrame с именами столбцов
    features = np.array([[total_square, rooms, floor, lat, lon, city, district]])
    features_df = pd.DataFrame(features, columns=['total_square', 'rooms', 'floor', 'lat', 'lon', 'city', 'district'])

    # Преобразуем категориальные данные в строки
    features_df['city'] = features_df['city'].astype(str)
    features_df['district'] = features_df['district'].astype(str)

    # Выполняем предсказание
    try:
        prediction = model.predict(features_df)[0]
        st.write(f"Прогнозируемая стоимость недвижимости: ${prediction:,.2f}")
    except Exception as e:
        st.write(f"Произошла ошибка при предсказании: {e}")
