from weather_service import Location, _location_rank


def test_major_city_ranks_ahead_of_small_namesake():
    capital = Location(
        name="北京",
        latitude=39.9075,
        longitude=116.39723,
        country="中国",
        admin1="北京市",
        admin2="北京市",
        country_code="CN",
        feature_code="PPLC",
        population=18_960_744,
    )
    namesake = Location(
        name="北京",
        latitude=30.72608,
        longitude=108.67483,
        country="中国",
        admin1="重庆市",
        admin2="重庆市",
        country_code="CN",
        feature_code="PPL",
    )

    ranked = sorted(
        [namesake, capital],
        key=lambda item: _location_rank(item, "北京"),
        reverse=True,
    )

    assert ranked[0] == capital
    assert "北京市" in ranked[0].display_name
    assert "39.907" in ranked[0].display_name
