"""
Machine learning and AI components for the weather forecast system.

- AIWeatherBriefing: natural-language weather summary generation (NLG)
- TemperaturePredictor: ARIMA time-series temperature forecast
- WeatherClusterer: K-Means clustering of daily weather patterns
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

#
# AI Weather Briefing Generator (NLG)
#

@dataclass(frozen=True)
class BriefingResult:
    summary: str
    highlights: list[str]
    outlook: str


def _pick(variants: list[str], seed: int = 0) -> str:
    return variants[seed % len(variants)]


def _temperature_feel(temp: float) -> str:
    if temp >= 35:
        return "酷热"
    if temp >= 30:
        return "炎热"
    if temp >= 25:
        return "温暖"
    if temp >= 18:
        return "舒适"
    if temp >= 10:
        return "偏凉"
    if temp >= 0:
        return "寒冷"
    return "严寒"


def _humidity_feel(rh: float) -> str:
    if rh >= 85:
        return "潮湿"
    if rh >= 60:
        return "较湿润"
    if rh >= 35:
        return "适中"
    return "干燥"






def generate_briefing(
    current: dict[str, Any],
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    seed: int | None = None,
) -> BriefingResult:
    if seed is None:
        seed = int(current.get("temperature_2m", 20) * 100) + len(hourly)
    temp = float(current["temperature_2m"])
    rh = float(current["relative_humidity_2m"])
    wind_speed = float(current["wind_speed_10m"])
    cloud = float(current["cloud_cover"])
    apparent = float(current["apparent_temperature"])
    precip = float(current.get("precipitation", 0))
    feel = _temperature_feel(temp)
    moisture = _humidity_feel(rh)
    sky_desc = "晴朗" if cloud < 30 else ("多云" if cloud < 70 else "阴沉")
    temp_detail = ""
    if abs(temp - apparent) >= 3:
        temp_detail = _pick(
            [f"，体感温度 {apparent:.0f}°C", f"（体感约 {apparent:.0f}°C）"],
             seed,
        )

    openers: list[str] = [
        f"当前气温 {temp:.1f}°C{temp_detail}，天气{sky_desc}，空气{moisture}。",
        f"此刻{sky_desc}，{feel}，气温 {temp:.1f}°C{temp_detail}，相对湿度 {rh:.0f}%。",
        f"{sky_desc}的天气，气温 {temp:.1f}°C{temp_detail}，体感{feel}，湿度{moisture}。",
    ]
    summary = openers[seed % len(openers)]
    highlights: list[str] = []
    if wind_speed >= 25:
        highlights.append(f"当前风速 {wind_speed:.0f} km/h，请注意防风。")
    if temp >= 35:
        highlights.append(f"高温天气：气温已达 {temp:.0f}°C，注意防暑降温。")
    if temp <= 0:
        highlights.append(f"寒冷天气：气温 {temp:.0f}°C，注意保暖防冻。")
    if rh >= 90:
        highlights.append(f"湿度较高（{rh:.0f}%），体感闷热，建议减少户外活动。")
    if not highlights:
        highlights.append(f"当前无明显极端天气风险，天气状况{feel}宜人。")
    outlook_rows = hourly.head(6)
    if len(outlook_rows) < 1:
        outlook = "短期数据不足，暂时无法生成展望。"
        return BriefingResult(summary=summary, highlights=highlights, outlook=outlook)
    temp_change = float(outlook_rows.iloc[-1]["temperature_2m"]) - float(
        outlook_rows.iloc[0]["temperature_2m"]
    )
    max_precip_prob = float(outlook_rows["precipitation_probability"].max())
    if abs(temp_change) < 1.5:
        trend = "保持平稳"
    elif temp_change > 0:
        trend = f"预计上升 {abs(temp_change):.1f}°C"
    else:
        trend = f"预计下降 {abs(temp_change):.1f}°C"
    rain_outlook = ""
    if max_precip_prob >= 50:
        rain_outlook = f"，降水概率最高可达 {max_precip_prob:.0f}%，建议携带雨具"
    multi_day = ""
    if len(daily) >= 2:
        avg1 = (float(daily.iloc[0]["temperature_2m_min"]) + float(daily.iloc[0]["temperature_2m_max"])) / 2
        avg2 = (float(daily.iloc[1]["temperature_2m_min"]) + float(daily.iloc[1]["temperature_2m_max"])) / 2
        if avg2 - avg1 > 2:
            multi_day = "预计未来两天将持续升温。"
        elif avg1 - avg2 > 2:
            multi_day = "预计未来两天将持续降温。"
        else:
            multi_day = "预计未来两天气温变化不大。"
    outlook = f"未来 6 小时气温{trend}{rain_outlook}。{multi_day}"
    return BriefingResult(summary=summary, highlights=highlights, outlook=outlook)


#
# ARIMA Time-Series Temperature Forecast
#


@dataclass(frozen=True)
class ForecastResult:
    aic: float
    order: tuple[int, int, int]
    historical_dates: list[str]
    historical_temps: list[float]
    forecast_dates: list[str]
    forecast_temps: list[float]
    forecast_lower: list[float]
    forecast_upper: list[float]
    history_day_count: int


def forecast_temperature(
    historical: dict[str, Any],
    *,
    forecast_steps: int = 5,
) -> ForecastResult | None:
    try:
        from statsmodels.tsa.arima.model import ARIMA  # type: ignore[import-untyped]
    except ImportError:
        return None
    dates = historical["daily"]["time"]
    temps = historical["daily"]["temperature_2m_mean"]
    if len(temps) < 20:
        return None
    best_aic = float("inf")
    best_order = (1, 0, 0)
    best_model = None
    for p in range(0, 4):
        for d in range(0, 2):
            for q in range(0, 4):
                if p == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(temps, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                        best_model = fitted
                except Exception:
                    continue
    if best_model is None:
        return None
    forecast = best_model.forecast(steps=forecast_steps)
    conf_int = best_model.get_forecast(steps=forecast_steps).conf_int()
    last_date = pd.to_datetime(dates[-1])
    forecast_dates_list = [
        (last_date + pd.Timedelta(days=i + 1)).strftime("%m-%d")
        for i in range(forecast_steps)
    ]
    display_days = min(14, len(dates))
    return ForecastResult(
        aic=best_aic,
        order=best_order,
        historical_dates=[d[-5:] for d in dates[-display_days:]],
        historical_temps=temps[-display_days:],
        forecast_dates=forecast_dates_list,
        forecast_temps=forecast.tolist(),
        forecast_lower=conf_int[:, 0].tolist(),
        forecast_upper=conf_int[:, 1].tolist(),
        history_day_count=len(dates),
    )


#
# Weather Pattern Clustering (K-Means)
#


@dataclass(frozen=True)
class ClusterResult:
    n_clusters: int
    labels: list[int]
    centroids: list[dict[str, float]]
    cluster_names: list[str]
    current_cluster: int
    current_cluster_name: str
    inertia: float
    elbow_data: list[tuple[int, float]]


def _daily_feature_matrix(daily: pd.DataFrame) -> np.ndarray | None:
    columns = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_probability_max",
        "uv_index_max",
    ]
    available = [c for c in columns if c in daily.columns]
    if len(available) < 3:
        return None
    data = daily[available].fillna(0).to_numpy(dtype=np.float64)
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    return (data - mins) / ranges


def _describe_cluster(centroid: dict[str, float]) -> tuple[str, str]:
    t_max = centroid.get("temperature_2m_max", 25)
    t_min = centroid.get("temperature_2m_min", 15)
    precip = centroid.get("precipitation_probability_max", 0)
    uv = centroid.get("uv_index_max", 3)
    avg_temp = (t_max + t_min) / 2
    if avg_temp > 33:
        temp_desc = "炎热"
    elif avg_temp > 25:
        temp_desc = "温暖"
    elif avg_temp > 15:
        temp_desc = "凉爽"
    elif avg_temp > 5:
        temp_desc = "寒冷"
    else:
        temp_desc = "严寒"
    moisture_desc = "多雨" if precip > 60 else ("偏湿" if precip > 30 else "干燥")
    sun_desc = "强日照" if uv > 6 else ("中等日照" if uv > 3 else "弱日照")
    name = f"{temp_desc}{moisture_desc}型"
    detail = f"均温 {avg_temp:.0f}°C, {moisture_desc}, {sun_desc}"
    return name, detail


def cluster_weather(daily: pd.DataFrame) -> ClusterResult | None:
    try:
        from sklearn.cluster import KMeans  # type: ignore[import-untyped]
    except ImportError:
        return None
    matrix = _daily_feature_matrix(daily)
    if matrix is None or len(matrix) < 3:
        return None
    column_names = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_probability_max",
        "uv_index_max",
    ]
    column_names = [c for c in column_names if c in daily.columns]
    n = min(max(2, len(matrix) - 1), 5)
    elbow_data: list[tuple[int, float]] = []
    max_k = min(n + 3, len(matrix))
    for k in range(1, max_k):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(matrix)
        elbow_data.append((k, float(km.inertia_)))
    model = KMeans(n_clusters=n, random_state=42, n_init=10)
    labels = model.fit_predict(matrix).tolist()
    centroids_raw = model.cluster_centers_
    centroids: list[dict[str, float]] = []
    cluster_names: list[str] = []
    for i in range(n):
        centroid_dict = dict(zip(column_names, centroids_raw[i].tolist()))
        for j, col in enumerate(column_names):
            raw = daily[col].fillna(0).to_numpy(dtype=np.float64)
            col_min, col_max = raw.min(), raw.max()
            rng = col_max - col_min
            if rng == 0:
                rng = 1.0
            centroid_dict[col] = round(centroid_dict[col] * rng + col_min, 1)
        centroids.append(centroid_dict)
        name, _detail = _describe_cluster(centroid_dict)
        cluster_names.append(name)
    current_cluster = labels[0]
    current_name = cluster_names[current_cluster]
    return ClusterResult(
        n_clusters=n,
        labels=labels,
        centroids=centroids,
        cluster_names=cluster_names,
        current_cluster=current_cluster,
        current_cluster_name=current_name,
        inertia=float(model.inertia_),
        elbow_data=elbow_data,
    )
