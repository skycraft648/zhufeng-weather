import pandas as pd

from weather_analysis import detect_changes, uv_level, wind_direction


def make_hourly(codes, rain=None, temperatures=None, gusts=None):
    size = len(codes)
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-06-12 08:00", periods=size, freq="h"),
            "weather_code": codes,
            "precipitation_probability": rain or [0] * size,
            "precipitation": [0] * size,
            "temperature_2m": temperatures or [25] * size,
            "wind_gusts_10m": gusts or [10] * size,
        }
    )


def test_detects_clear_to_rain_transition():
    frame = make_hourly(
        [0, 1, 2, 61, 61],
        rain=[0, 10, 30, 70, 80],
    )
    messages = detect_changes(frame)
    assert any("转为小雨" in message for message in messages)
    assert any("降水概率升至 70%" in message for message in messages)


def test_detects_temperature_drop_and_wind():
    frame = make_hourly(
        [2, 2, 2, 2],
        temperatures=[30, 29, 28, 26],
        gusts=[20, 30, 55, 40],
    )
    messages = detect_changes(frame)
    assert any("下降约 4.0℃" in message for message in messages)
    assert any("注意防风" in message for message in messages)


def test_reports_stable_weather():
    messages = detect_changes(make_hourly([1, 1, 1, 1]))
    assert messages == ["未来 12 小时天气总体稳定，暂未发现明显变化。"]


def test_helpers():
    assert uv_level(7) == "高"
    assert wind_direction(90) == "东"

