from unittest.mock import Mock, patch

from weather_service import search_locations


@patch("weather_service._search_nominatim", return_value=[])
@patch("weather_service.requests.get")
def test_district_search_retries_without_suffix(mock_get, _mock_nominatim):
    empty = Mock()
    empty.raise_for_status.return_value = None
    empty.json.return_value = {}

    found = Mock()
    found.raise_for_status.return_value = None
    found.json.return_value = {
        "results": [
            {
                "name": "海淀",
                "latitude": 39.99064,
                "longitude": 116.28868,
                "feature_code": "PPLA3",
                "country_code": "CN",
                "country": "中国",
                "admin1": "北京市",
                "admin2": "北京市",
            }
        ]
    }
    mock_get.side_effect = [empty, found]

    results = search_locations("海淀区")

    assert results[0].name == "海淀"
    assert mock_get.call_args_list[1].kwargs["params"]["name"] == "海淀"


@patch("weather_service._search_location_payload", return_value=[])
@patch("weather_service._search_nominatim")
def test_uses_osm_fallback_for_missing_district(mock_nominatim, _mock_open_meteo):
    from weather_service import Location

    mock_nominatim.return_value = [
        Location(
            name="余杭区",
            latitude=30.2762431,
            longitude=119.9746471,
            country="中国",
            admin1="浙江省",
            admin2="杭州市",
            country_code="CN",
            feature_code="OSM",
        )
    ]

    results = search_locations("杭州市余杭区")

    assert results[0].name == "余杭区"
    assert results[0].admin2 == "杭州市"
