"""Tests for the machine learning / AI weather components."""
from __future__ import annotations
import pandas as pd
import pytest
from weather_ml import (
    BriefingResult,
    generate_briefing,
    ForecastResult,
    forecast_temperature,
    _temperature_feel,
    _humidity_feel,
    cluster_weather,
    _daily_feature_matrix,
    _describe_cluster,
)
def make_current(temp=22.0, rh=55, wind=10, cloud=30, apparent=21.0, precip=0.0):
    return {
        "temperature_2m": temp,
        "relative_humidity_2m": rh,
        "wind_speed_10m": wind,
        "cloud_cover": cloud,
        "apparent_temperature": apparent,
        "precipitation": precip,
    }
def make_hourly_df(temps=None, size=24):
    temps = temps or [20 + i * 0.5 for i in range(size)]
    return pd.DataFrame({
        "time": pd.date_range("2026-06-12 08:00", periods=size, freq="h"),
        "temperature_2m": temps,
        "relative_humidity_2m": [55] * size,
        "apparent_temperature": [t - 1 for t in temps],
        "precipitation_probability": [10] * size,
        "precipitation": [0] * size,
        "weather_code": [1] * size,
        "cloud_cover": [30] * size,
        "visibility": [10000] * size,
        "wind_speed_10m": [10] * size,
        "wind_gusts_10m": [15] * size,
        "uv_index": [3] * size,
    })
def make_daily_df(temps_max=None, temps_min=None, size=3):
    if temps_max is None or len(temps_max) != size:
        temps_max = [28 - i * 0.5 for i in range(size)]
    if temps_min is None or len(temps_min) != size:
        temps_min = [18 - i * 0.5 for i in range(size)]
    return pd.DataFrame({
        "time": pd.date_range("2026-06-12", periods=size, freq="D"),
        "weather_code": [1] * size,
        "temperature_2m_max": temps_max,
        "temperature_2m_min": temps_min,
        "precipitation_probability_max": [20, 10, 60] if size <= 3 else [20 + i * 5 for i in range(size)],
        "precipitation_probability_max": [20, 10, 60][:size] if size <= 3 else [20 + i * 5 for i in range(size)],
        "uv_index_max": [5, 6, 2][:size] if size <= 3 else [3 + i for i in range(size)],
    })
#
# TestHelpers
#
class TestHelpers:
    def test_temperature_feel_ranges(self):
        assert _temperature_feel(36) in ("酷热", "炎热")
        assert _temperature_feel(30) in ("炎热", "温暖")
        assert "暖" in _temperature_feel(25)
        assert "适" in _temperature_feel(18)
        assert "凉" in _temperature_feel(10)
        assert _temperature_feel(0) in ("寒冷", "严寒")
        assert _temperature_feel(-5) == "严寒"
    def test_humidity_feel_ranges(self):
        assert _humidity_feel(90) == "潮湿"
        assert _humidity_feel(60) in ("较湿润", "适中")
        assert _humidity_feel(35) == "适中"
        assert _humidity_feel(20) == "干燥"#
# TestBriefing
#
class TestBriefing:
    def test_generates_briefing_basic(self):
        current = make_current(22, 55, 10, 20, 21)
        hourly = make_hourly_df()
        daily = make_daily_df()
        result = generate_briefing(current, hourly, daily)
        assert isinstance(result, BriefingResult)
        assert len(result.summary) > 0
        assert len(result.highlights) > 0
        assert len(result.outlook) > 0
    def test_high_temp_triggers_alert(self):
        current = make_current(36, 40, 5, 20, 35)
        hourly = make_hourly_df()
        daily = make_daily_df()
        result = generate_briefing(current, hourly, daily)
        assert any("高温" in h or "暑" in h for h in result.highlights)
    def test_cold_temp_triggers_alert(self):
        current = make_current(-3, 50, 5, 20, -5)
        hourly = make_hourly_df()
        daily = make_daily_df()
        result = generate_briefing(current, hourly, daily)
        assert any("寒冷" in h or "冻" in h for h in result.highlights)
    def test_high_wind_triggers_alert(self):
        current = make_current(22, 50, 30, 20, 21)
        hourly = make_hourly_df()
        daily = make_daily_df()
        result = generate_briefing(current, hourly, daily)
        assert any("风" in h for h in result.highlights)
    def test_seed_determinism(self):
        current = make_current()
        hourly = make_hourly_df()
        daily = make_daily_df()
        r1 = generate_briefing(current, hourly, daily, seed=42)
        r2 = generate_briefing(current, hourly, daily, seed=42)
        assert r1.summary == r2.summary
        assert r1.outlook == r2.outlook
    def test_empty_hourly_handled(self):
        current = make_current()
        hourly = pd.DataFrame()
        daily = make_daily_df()
        result = generate_briefing(current, hourly, daily)
        assert isinstance(result, BriefingResult)
#
# TestTemperaturePredictor
#
class TestTemperaturePredictor:
    def test_forecasts_with_valid_historical_data(self):
        historical = {
            "daily": {
                "time": [f"2026-06-{d:02d}" for d in range(1, 31)],
                "temperature_2m_mean": [22.0 + i * 0.3 for i in range(30)],
            }
        }
        result = forecast_temperature(historical, forecast_steps=3)
        if result is None:
            pytest.skip("statsmodels not installed")
        assert isinstance(result, ForecastResult)
        assert result.history_day_count == 30
        assert len(result.forecast_temps) == 3
        assert len(result.forecast_lower) == 3
        assert len(result.forecast_upper) == 3
        assert result.order[0] >= 0
    def test_insufficient_historical_data_returns_none(self):
        historical = {
            "daily": {
                "time": [f"2026-06-{d:02d}" for d in range(1, 6)],
                "temperature_2m_mean": [22.0] * 5,
            }
        }
        result = forecast_temperature(historical)
        assert result is None
    def test_forecast_result_contains_required_fields(self):
        historical = {
            "daily": {
                "time": [f"2026-06-{d:02d}" for d in range(1, 31)],
                "temperature_2m_mean": [22.0 + i * 0.2 for i in range(30)],
            }
        }
        result = forecast_temperature(historical, forecast_steps=5)
        if result is None:
            pytest.skip("statsmodels not installed")
        assert len(result.historical_dates) > 0
        assert len(result.forecast_dates) == 5
        assert result.aic > -1e9
#
# TestWeatherClusterer
#
class TestWeatherClusterer:
    def test_clusters_valid_daily_data(self):
        daily = make_daily_df(size=7)
        result = cluster_weather(daily)
        if result is None:
            pytest.skip("scikit-learn not installed")
        assert result.n_clusters >= 2
        assert len(result.labels) == 7
        assert len(result.centroids) == result.n_clusters
        assert len(result.cluster_names) == result.n_clusters
        assert result.current_cluster >= 0
        assert len(result.elbow_data) >= 1
    def test_current_cluster_is_first_day(self):
        daily = make_daily_df(size=7)
        result = cluster_weather(daily)
        if result is None:
            pytest.skip("scikit-learn not installed")
        assert result.current_cluster == result.labels[0]
    def test_insufficient_data_returns_none(self):
        daily = make_daily_df(size=2)
        result = cluster_weather(daily)
        assert result is None
    def test_feature_matrix_normalisation(self):
        daily = make_daily_df(size=5)
        matrix = _daily_feature_matrix(daily)
        assert matrix is not None
        assert matrix.shape == (5, 4)
        assert matrix.max() <= 1.0 and matrix.min() >= 0.0
    def test_describe_cluster_output(self):
        centroid = {
            "temperature_2m_max": 35.0,
            "temperature_2m_min": 28.0,
            "precipitation_probability_max": 10.0,
            "uv_index_max": 8.0,
        }
        name, detail = _describe_cluster(centroid)
        assert isinstance(name, str)
        assert len(name) > 0
        assert isinstance(detail, str)
