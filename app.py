import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

month_to_season = {12: "winter", 1: "winter", 2: "winter",
                   3: "spring", 4: "spring", 5: "spring",
                   6: "summer", 7: "summer", 8: "summer",
                   9: "autumn", 10: "autumn", 11: "autumn"}


@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["season"] = data["timestamp"].dt.month.map(lambda x: month_to_season[x])
    return data


def seasonal_analysis(data):
    seasonal_stats = (
        data.groupby(["city", "season"])["temperature"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "seasonal_mean", "std": "seasonal_std"})
    )
    return seasonal_stats


def check_temperature_anomaly(city, current_temp, seasonal_stats):
    season = pd.Timestamp.now().month
    current_season = month_to_season[season]
    city_stats = seasonal_stats[(seasonal_stats["city"] == city) & (seasonal_stats["season"] == current_season)]
    if not city_stats.empty:
        mean_temp = city_stats["seasonal_mean"].values[0]
        std_temp = city_stats["seasonal_std"].values[0]
        lower_bound = mean_temp - 2 * std_temp
        upper_bound = mean_temp + 2 * std_temp
        is_anomaly = not (lower_bound <= current_temp <= upper_bound)
        return is_anomaly, lower_bound, upper_bound
    return None, None, None


def get_current_temperature(city, api_key):
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data["main"]["temp"], None
    elif response.status_code == 401:
        return None, "Invalid API key. Please see https://openweathermap.org/faq#error401 for more info."
    else:
        return None, f"Error: {response.status_code}, {response.text}"


st.title("Анализ погоды")

uploaded_file = st.file_uploader("Загрузите CSV-файл с историческими данными", type=["csv"])
if uploaded_file:
    data = load_data(uploaded_file)
    cities = sorted(data["city"].unique())
    city = st.selectbox("Выберите город", cities)

    seasonal_stats = seasonal_analysis(data)
    city_data = data[data["city"] == city]

    st.subheader("Исторические данные для выбранного города")
    st.write(city_data.describe())

    st.subheader("Временной ряд температур с выделением аномалий")
    city_data["rolling_mean"] = city_data["temperature"].rolling(window=30).mean()

    mean_temp = city_data["temperature"].mean()
    std_temp = city_data["temperature"].std()
    lower_bound = mean_temp - 2 * std_temp
    upper_bound = mean_temp + 2 * std_temp

    plt.figure(figsize=(10, 5))
    plt.plot(city_data["timestamp"], city_data["temperature"], label="Температура", alpha=0.5)
    plt.plot(city_data["timestamp"], city_data["rolling_mean"], label="Скользящее среднее (30 дней)", color="orange")
    plt.fill_between(
        city_data["timestamp"], lower_bound, upper_bound, color="green", alpha=0.2, label="Интервал (норма)"
    )

    anomalies = city_data[
        (city_data["temperature"] > upper_bound) | (city_data["temperature"] < lower_bound)
        ]
    plt.scatter(anomalies["timestamp"], anomalies["temperature"], color="red", label="Аномалии")

    plt.xlabel("Дата")
    plt.ylabel("Температура")
    plt.title(f"Температурный ряд для {city}")
    plt.legend()
    st.pyplot(plt)

    st.subheader("Сезонный профиль")
    city_seasonal_stats = seasonal_stats[seasonal_stats["city"] == city]
    plt.figure(figsize=(10, 5))
    plt.errorbar(
        city_seasonal_stats["season"],
        city_seasonal_stats["seasonal_mean"],
        yerr=city_seasonal_stats["seasonal_std"],
        fmt="o", capsize=5, label="Среднее ± Стандартное отклонение"
    )
    plt.xlabel("Сезон")
    plt.ylabel("Температура")
    plt.legend()
    st.pyplot(plt)

    st.subheader("Проверка текущей температуры")
    api_key = st.text_input("Введите API-ключ OpenWeatherMap")
    if api_key:
        current_temp, error = get_current_temperature(city, api_key)
        if error:
            st.error(error)
        elif current_temp is not None:
            is_anomaly, lower_bound, upper_bound = check_temperature_anomaly(city, current_temp, seasonal_stats)
            st.write(f"Текущая температура в {city}: {current_temp}°C")
            if is_anomaly:
                st.warning(f"Температура аномальная! (норма: {lower_bound:.2f}°C — {upper_bound:.2f}°C)")
            else:
                st.success("Температура в пределах нормы.")
