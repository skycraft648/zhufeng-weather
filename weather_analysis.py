"""Rule-based short-term weather change detection."""

from __future__ import annotations

import pandas as pd

from weather_codes import weather_info


def uv_level(value: float) -> str:
    if value < 3:
        return "低"
    if value < 6:
        return "中等"
    if value < 8:
        return "高"
    if value < 11:
        return "很高"
    return "极高"


def wind_direction(degrees: float) -> str:
    directions = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
    return directions[round(float(degrees) / 45) % 8]


def detect_changes(hourly: pd.DataFrame, limit_hours: int = 12) -> list[str]:
    """Describe significant changes expected during the next several hours."""
    future = hourly.head(limit_hours + 1).copy()
    if len(future) < 2:
        return ["逐小时数据不足，暂时无法判断天气变化。"]

    messages: list[str] = []
    start_time = future.iloc[0]["time"]
    start_code = int(future.iloc[0]["weather_code"])
    start_category = weather_info(start_code)[2]

    impactful = {"rain", "storm", "snow", "fog"}
    transition_row = None
    for _, row in future.iloc[1:].iterrows():
        category = weather_info(int(row["weather_code"]))[2]
        if category != start_category and (
            category in impactful or start_category in impactful
        ):
            transition_row = row
            break

    if transition_row is not None:
        hours = max(
            1, round((transition_row["time"] - start_time).total_seconds() / 3600)
        )
        description = weather_info(int(transition_row["weather_code"]))[0]
        messages.append(f"约 {hours} 小时后天气可能转为{description}。")

    rain_rows = future[
        (future["precipitation_probability"].fillna(0) >= 50)
        | (future["precipitation"].fillna(0) >= 0.1)
    ]
    if start_category not in {"rain", "storm", "snow"} and not rain_rows.empty:
        row = rain_rows.iloc[0]
        hours = max(1, round((row["time"] - start_time).total_seconds() / 3600))
        probability = int(round(row["precipitation_probability"]))
        messages.append(f"约 {hours} 小时后降水概率升至 {probability}%，建议留意雨具。")

    comparison_index = min(3, len(future) - 1)
    temperature_delta = (
        float(future.iloc[comparison_index]["temperature_2m"])
        - float(future.iloc[0]["temperature_2m"])
    )
    if abs(temperature_delta) >= 3:
        trend = "升高" if temperature_delta > 0 else "下降"
        messages.append(
            f"未来 {comparison_index} 小时气温预计{trend}约 {abs(temperature_delta):.1f}℃。"
        )

    max_gust = float(future["wind_gusts_10m"].fillna(0).max())
    if max_gust >= 50:
        messages.append(f"未来 {limit_hours} 小时最大阵风约 {max_gust:.0f} km/h，注意防风。")

    if not messages:
        messages.append(f"未来 {limit_hours} 小时天气总体稳定，暂未发现明显变化。")
    return messages
