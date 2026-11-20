"""Open-Meteo API client and response normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


class WeatherServiceError(RuntimeError):
    """Raised when weather data cannot be obtained or parsed."""


@dataclass(frozen=True)
class Location:
    name: str
    latitude: float
    longitude: float
    country: str = ""
    admin1: str = ""
    admin2: str = ""
    country_code: str = ""
    feature_code: str = ""
    population: int = 0
    timezone: str = "auto"

    @property
    def display_name(self) -> str:
        parts = [self.name, self.admin2, self.admin1, self.country]
        place = "，".join(dict.fromkeys(part for part in parts if part))
        return f"{place}（{self.latitude:.3f}, {self.longitude:.3f}）"


def _location_rank(location: Location, query: str) -> tuple[int, int, int, int, int]:
    """Rank exact, major and populated places ahead of small namesakes."""
    normalized_query = query.strip().casefold()
    normalized_name = location.name.strip().casefold()
    exact_name = int(normalized_name == normalized_query)
    name_in_query = int(normalized_name in normalized_query)
    context_match = sum(
        int(part.strip().casefold().rstrip("省市区县") in normalized_query)
        for part in (location.admin1, location.admin2)
        if part.strip().rstrip("省市区县")
    )
    feature_priority = {
        "PPLC": 5,  # national capital
        "PPLA": 4,  # first-order administrative capital
        "PPLA2": 3,
        "PPLA3": 2,
        "PPL": 1,
    }.get(location.feature_code, 0)
    domestic = int(location.country_code == "CN")
    return exact_name, name_in_query + context_match, domestic, feature_priority, location.population


def _search_location_payload(query: str, count: int) -> list[dict[str, Any]]:
    response = requests.get(
        GEOCODING_URL,
        params={
            "name": query,
            "count": count,
            "language": "zh",
            "format": "json",
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json().get("results", [])


def _search_nominatim(query: str, count: int) -> list[Location]:
    response = requests.get(
        NOMINATIM_URL,
        params={
            "q": query,
            "format": "jsonv2",
            "addressdetails": 1,
            "limit": min(count, 10),
            "accept-language": "zh-CN",
            "countrycodes": "cn",
        },
        headers={"User-Agent": "WeatherForecastStudentProject/1.0"},
        timeout=12,
    )
    response.raise_for_status()

    locations = []
    for item in response.json():
        address = item.get("address", {})
        display_parts = [
            part.strip() for part in item.get("display_name", "").split(",") if part.strip()
        ]
        admin2 = (
            address.get("city")
            or address.get("county")
            or address.get("municipality")
            or ""
        )
        if admin2 == item.get("name") and len(display_parts) > 1:
            admin2 = display_parts[1]
        admin1 = address.get("state") or address.get("province") or ""
        locations.append(
            Location(
                name=item.get("name") or item.get("display_name", "").split(",")[0],
                latitude=float(item["lat"]),
                longitude=float(item["lon"]),
                country=address.get("country", "中国"),
                admin1=admin1,
                admin2=admin2,
                country_code=address.get("country_code", "").upper(),
                feature_code="OSM",
                population=int(float(item.get("importance", 0)) * 1_000_000),
            )
        )
    return locations


def search_locations(query: str, count: int = 20) -> list[Location]:
    """Search matching places by Chinese or international place name."""
    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    try:
        results = _search_location_payload(cleaned_query, count)
        if not results and cleaned_query.endswith(("区", "县", "市")):
            results = _search_location_payload(cleaned_query[:-1], count)
    except (requests.RequestException, ValueError) as exc:
        raise WeatherServiceError(f"地点搜索失败：{exc}") from exc

    try:
        osm_locations = _search_nominatim(cleaned_query, count)
    except (requests.RequestException, ValueError):
        osm_locations = []

    locations = [
        Location(
            name=item["name"],
            latitude=float(item["latitude"]),
            longitude=float(item["longitude"]),
            country=item.get("country", ""),
            admin1=item.get("admin1", ""),
            admin2=item.get("admin2", ""),
            country_code=item.get("country_code", ""),
            feature_code=item.get("feature_code", ""),
            population=int(item.get("population") or 0),
            timezone=item.get("timezone", "auto"),
        )
        for item in results
    ]
    combined = locations + osm_locations
    unique_locations = {}
    for location in combined:
        coordinate_key = (round(location.latitude, 3), round(location.longitude, 3))
        unique_locations.setdefault(coordinate_key, location)

    return sorted(
        unique_locations.values(),
        key=lambda item: _location_rank(item, cleaned_query),
        reverse=True,
    )


def fetch_weather(location: Location, forecast_days: int = 3) -> dict[str, Any]:
    """Fetch current, hourly and daily weather for a location."""
    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "timezone": "auto",
        "forecast_days": forecast_days,
        "current": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "is_day",
                "precipitation",
                "weather_code",
                "cloud_cover",
                "wind_speed_10m",
                "wind_direction_10m",
            ]
        ),
        "hourly": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "precipitation_probability",
                "precipitation",
                "weather_code",
                "cloud_cover",
                "visibility",
                "wind_speed_10m",
                "wind_gusts_10m",
                "uv_index",
            ]
        ),
        "daily": ",".join(
            [
                "weather_code",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_probability_max",
                "uv_index_max",
                "sunrise",
                "sunset",
            ]
        ),
    }

    try:
        response = requests.get(FORECAST_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        if "current" not in data or "hourly" not in data:
            raise WeatherServiceError("天气接口返回的数据不完整。")
        return data
    except requests.RequestException as exc:
        raise WeatherServiceError(f"天气数据获取失败：{exc}") from exc
    except ValueError as exc:
        raise WeatherServiceError("天气接口返回了无法解析的数据。") from exc


def hourly_dataframe(data: dict[str, Any], hours: int = 48) -> pd.DataFrame:
    """Normalize hourly API arrays and keep rows from the current hour."""
    frame = pd.DataFrame(data["hourly"])
    frame["time"] = pd.to_datetime(frame["time"])
    current_time = pd.to_datetime(data["current"]["time"]).floor("h")
    frame = frame.loc[frame["time"] >= current_time].head(hours).copy()
    return frame.reset_index(drop=True)


def daily_dataframe(data: dict[str, Any]) -> pd.DataFrame:
    """Normalize daily API arrays."""
    frame = pd.DataFrame(data["daily"])
    frame["time"] = pd.to_datetime(frame["time"])
    return frame


def fetch_historical(
    location: Location, *, days: int = 90
) -> dict[str, Any] | None:
    """Fetch historical daily weather from Open-Meteo Archive API.

    Returns daily temperature data for ARIMA training. The archive
    endpoint is free and does not require an API key.
    """
    import datetime as _dt
    end_date = _dt.date.today() - _dt.timedelta(days=1)
    start_date = end_date - _dt.timedelta(days=days)
    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": "temperature_2m_mean",
        "timezone": "auto",
    }
    try:
        response = requests.get(ARCHIVE_URL, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        if "daily" not in data:
            raise WeatherServiceError("历史天气数据不完整。")
        return data
    except requests.RequestException as exc:
        raise WeatherServiceError(f"历史数据获取失败：{exc}") from exc
    except ValueError as exc:
        raise WeatherServiceError("历史数据无法解析。") from exc
