"""WMO weather code descriptions used by Open-Meteo."""

WEATHER_CODES = {
    0: ("晴朗", "☀️", "clear"),
    1: ("大致晴朗", "🌤️", "clear"),
    2: ("局部多云", "⛅", "cloudy"),
    3: ("阴天", "☁️", "cloudy"),
    45: ("有雾", "🌫️", "fog"),
    48: ("雾凇", "🌫️", "fog"),
    51: ("小毛毛雨", "🌦️", "rain"),
    53: ("毛毛雨", "🌦️", "rain"),
    55: ("强毛毛雨", "🌧️", "rain"),
    56: ("轻微冻毛毛雨", "🌧️", "rain"),
    57: ("强冻毛毛雨", "🌧️", "rain"),
    61: ("小雨", "🌦️", "rain"),
    63: ("中雨", "🌧️", "rain"),
    65: ("大雨", "🌧️", "rain"),
    66: ("轻微冻雨", "🌧️", "rain"),
    67: ("强冻雨", "🌧️", "rain"),
    71: ("小雪", "🌨️", "snow"),
    73: ("中雪", "🌨️", "snow"),
    75: ("大雪", "❄️", "snow"),
    77: ("米雪", "🌨️", "snow"),
    80: ("小阵雨", "🌦️", "rain"),
    81: ("中阵雨", "🌧️", "rain"),
    82: ("强阵雨", "⛈️", "rain"),
    85: ("小阵雪", "🌨️", "snow"),
    86: ("强阵雪", "❄️", "snow"),
    95: ("雷暴", "⛈️", "storm"),
    96: ("雷暴伴小冰雹", "⛈️", "storm"),
    99: ("雷暴伴大冰雹", "⛈️", "storm"),
}


def weather_info(code: int) -> tuple[str, str, str]:
    """Return Chinese description, icon and broad weather category."""
    return WEATHER_CODES.get(int(code), ("未知天气", "🌡️", "unknown"))

