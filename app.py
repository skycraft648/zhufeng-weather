"""Streamlit weather forecast dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from weather_analysis import detect_changes, uv_level, wind_direction
from weather_ml import (
    cluster_weather,
    ForecastResult,
    forecast_temperature,
    generate_briefing,
)
from weather_codes import weather_info
from weather_service import (
    Location,
    WeatherServiceError,
    daily_dataframe,
    fetch_weather,
    fetch_historical,
    hourly_dataframe,
    search_locations,
)


st.set_page_config(page_title="逐风天气", page_icon="🌤️", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem;}
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #0d9488 0%, #14b8a6 50%, #22d3ee 100%);
        border: none; color: white;
        padding: 14px; border-radius: 16px;
    }
    [data-testid="stMetric"] * {
        color: white !important;
    }
    .weather-hero {
        padding: 24px; border-radius: 22px; color: white;
        background: linear-gradient(125deg, #2267b4 0%, #52a5df 58%, #87d6dc 100%);
        box-shadow: 0 12px 35px rgba(28, 92, 150, .18);
        margin-bottom: 18px;
    }
    .weather-hero h2 {margin: 0 0 6px 0;}
    .weather-hero p {margin: 4px 0; opacity: .94;}
    .change-card {
        background: linear-gradient(135deg, #c084fc 0%, #e9d5ff 50%, #ffffff 100%); border-left: 5px solid #7c3aed; color: #4c1d95;
        border-radius: 10px; padding: 12px 16px; margin: 8px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_location_search_v2(query: str) -> list[Location]:
    return search_locations(query)


@st.cache_data(ttl=600, show_spinner=False)
def cached_weather(location: Location) -> dict:
    return fetch_weather(location)


def weather_chart(hourly: pd.DataFrame) -> go.Figure:
    display = hourly.head(24)
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=display["time"],
            y=display["temperature_2m"],
            name="气温",
            mode="lines+markers",
            line={"color": "#ef6c45", "width": 3},
            hovertemplate="%{x|%H:%M}<br>%{y:.1f}℃<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Bar(
            x=display["time"],
            y=display["precipitation_probability"],
            name="降水概率",
            marker_color="rgba(55, 137, 204, .42)",
            hovertemplate="%{x|%H:%M}<br>%{y:.0f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.update_yaxes(title_text="气温 (℃)", secondary_y=False)
    figure.update_yaxes(title_text="降水概率 (%)", range=[0, 100], secondary_y=True)
    figure.update_layout(
        height=390,
        margin={"l": 15, "r": 15, "t": 20, "b": 15},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.1},
        bargap=0.45,
    )
    return figure


def render_dashboard(location: Location, data: dict) -> None:
    current = data["current"]
    hourly = hourly_dataframe(data)
    daily = daily_dataframe(data)
    description, icon, _ = weather_info(current["weather_code"])
    current_uv = float(hourly.iloc[0]["uv_index"])

    st.markdown(
        f"""
        <div class="weather-hero">
          <h2>{icon} {location.display_name}</h2>
          <p>{description} · 数据更新时间 {pd.to_datetime(current["time"]).strftime("%Y-%m-%d %H:%M")}</p>
          <p style="font-size: 2.5rem; font-weight: 700;">{current["temperature_2m"]:.1f}℃</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(6)
    columns[0].metric("体感温度", f"{current['apparent_temperature']:.1f}℃")
    columns[1].metric("相对湿度", f"{current['relative_humidity_2m']:.0f}%")
    columns[2].metric("紫外线", f"{current_uv:.1f}", uv_level(current_uv))
    columns[3].metric("当前降水", f"{current['precipitation']:.1f} mm")
    columns[4].metric("云量", f"{current['cloud_cover']:.0f}%")
    columns[5].metric(
        "风速",
        f"{current['wind_speed_10m']:.1f} km/h",
        f"{wind_direction(current['wind_direction_10m'])}风",
    )

    st.subheader("短时天气变化")
    for message in detect_changes(hourly):
        st.markdown(f'<div class="change-card">{message}</div>', unsafe_allow_html=True)

    st.subheader("未来 24 小时")
    st.plotly_chart(weather_chart(hourly), use_container_width=True)

    table = hourly.head(24).copy()
    table["时间"] = table["time"].dt.strftime("%m-%d %H:%M")
    table["天气"] = table["weather_code"].map(lambda code: " ".join(weather_info(code)[:2]))
    table["气温"] = table["temperature_2m"].map(lambda value: f"{value:.1f}℃")
    table["湿度"] = table["relative_humidity_2m"].map(lambda value: f"{value:.0f}%")
    table["降水概率"] = table["precipitation_probability"].map(lambda value: f"{value:.0f}%")
    table["紫外线"] = table["uv_index"].map(lambda value: f"{value:.1f}")
    with st.expander("查看逐小时详情"):
        st.dataframe(
            table[["时间", "天气", "气温", "湿度", "降水概率", "紫外线"]],
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("未来 3 天")
    day_columns = st.columns(len(daily))
    for column, (_, row) in zip(day_columns, daily.iterrows()):
        text, day_icon, _ = weather_info(row["weather_code"])
        with column:
            st.markdown(f"**{row['time'].strftime('%m月%d日')}**")
            st.markdown(f"### {day_icon} {text}")
            st.write(f"{row['temperature_2m_min']:.0f}℃ ~ {row['temperature_2m_max']:.0f}℃")
            st.caption(
                f"降水 {row['precipitation_probability_max']:.0f}% · "
                f"UV {row['uv_index_max']:.1f}"
            )


st.title("逐风天气")
st.caption("输入城市、区县或经纬度，查看实时天气和未来数小时的变化趋势。")

with st.form("location_form"):
    query = st.text_input(
        "地点",
        value="上海",
        placeholder="例如：海淀区、长兴县、余杭区",
        help="区县搜索不到时，系统会自动尝试去掉“区、县、市”后缀。",
    )
    submitted = st.form_submit_button("查询天气", type="primary", use_container_width=True)

with st.expander("使用经纬度精确定位"):
    coordinate_columns = st.columns(2)
    latitude = coordinate_columns[0].number_input(
        "纬度", min_value=-90.0, max_value=90.0, value=31.2304, format="%.6f"
    )
    longitude = coordinate_columns[1].number_input(
        "经度", min_value=-180.0, max_value=180.0, value=121.4737, format="%.6f"
    )
    coordinate_name = st.text_input("地点名称（可选）", value="自定义位置")
    coordinate_submitted = st.button("查询此坐标", use_container_width=True)

if coordinate_submitted:
    st.session_state.location_results = [
        Location(
            name=coordinate_name.strip() or "自定义位置",
            latitude=latitude,
            longitude=longitude,
            timezone="auto",
        )
    ]
    st.session_state.selected_location = 0

if (submitted or "selected_location" not in st.session_state) and not coordinate_submitted:
    try:
        with st.spinner("正在搜索地点..."):
            results = cached_location_search_v2(query)
        if not results:
            st.warning("没有找到该地点，请尝试输入更完整的城市或地区名称。")
            st.stop()
        st.session_state.location_results = results
        st.session_state.selected_location = 0
    except WeatherServiceError as exc:
        st.error(str(exc))
        st.stop()

results = st.session_state.get("location_results", [])
if results:
    selected_index = st.selectbox(
        "选择匹配地点",
        options=range(len(results)),
        format_func=lambda index: results[index].display_name,
        index=st.session_state.get("selected_location", 0),
    )
    st.session_state.selected_location = selected_index
    selected = results[selected_index]

    try:
        with st.spinner("正在获取最新天气..."):
            weather_data = cached_weather(selected)
        render_dashboard(selected, weather_data)

        # ---- AI 智能分析 ----
        st.divider()
        st.subheader("AI 智能分析")
        hourly_df = hourly_dataframe(weather_data)
        daily_df = daily_dataframe(weather_data)
        current_data = weather_data["current"]
        ai_tab1, ai_tab2, ai_tab3 = st.tabs(["AI 天气简报", "ML 气温预测", "天气模式聚类"])

        with ai_tab1:
            st.caption("由 NLG 引擎自动生成的自然语言天气简报")
            try:
                briefing = generate_briefing(current_data, hourly_df, daily_df)
                if briefing is None:
                    raise RuntimeError("generate_briefing returned None")
                st.markdown(
                    f'<div style="background:linear-gradient(180deg,#ffffff 0%,#ffedd5 50%,#fed7aa 100%);border-radius:14px;padding:22px 24px;border:1px solid #f97316;color:#7c2d12;"><p style="font-size:1.08rem;line-height:1.85;margin:0;">{briefing.summary}</p></div>',
                    unsafe_allow_html=True,
                )
                if briefing.highlights:
                    for h in briefing.highlights:
                        st.markdown(f'<div style="background:linear-gradient(180deg,#ffffff 0%,#e5e7eb 100%);border-left:5px solid #6b7280;border-radius:8px;padding:10px 14px;margin:6px 0;color:#374151;">{h}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div style="background:linear-gradient(180deg,#ffffff 0%,#ccfbf1 50%,#99f6e4 100%);border-radius:14px;padding:18px 24px;border:1px solid #14b8a6;margin-top:12px;color:#134e4a;"><p style="font-size:1rem;line-height:1.75;margin:0;">{briefing.outlook}</p></div>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"简报失败：{e}")

        with ai_tab2:
            st.caption("基于 ARIMA 时间序列模型的气温预测")
            try:
                with st.spinner("正在获取历史数据并训练 ARIMA 模型..."):
                    hist_data = fetch_historical(selected)
            except WeatherServiceError:
                hist_data = None
            if hist_data is None:
                st.warning("无法获取历史天气数据，请检查网络后重试。")
            else:
                forecast_result = forecast_temperature(hist_data, forecast_steps=5)
                if forecast_result is None:
                    st.warning("statsmodels 未安装，请运行 pip install statsmodels 后重试。")
                else:
                    p, d, q = forecast_result.order
                    col1, col2 = st.columns(2)
                    col1.metric("ARIMA 模型", f"({p},{d},{q})")
                    col2.metric("AIC", f"{forecast_result.aic:.1f}")
                    st.caption(f"基于 {forecast_result.history_day_count} 天历史数据训练，预测未来 5 天")
                    chart_df = pd.DataFrame({
                        "历史气温": forecast_result.historical_temps,
                    }, index=forecast_result.historical_dates)
                    st.line_chart(chart_df, height=220)
                    st.caption("未来 5 天气温预测（含 95% 置信区间）")
                    forecast_lines = ""
                    for i, (date, temp, lo, hi) in enumerate(zip(
                        forecast_result.forecast_dates,
                        forecast_result.forecast_temps,
                        forecast_result.forecast_lower,
                        forecast_result.forecast_upper,
                    )):
                        forecast_lines += f"<b>{date}</b>: {temp:.1f}°C (95% CI: {lo:.1f} ~ {hi:.1f})&nbsp;&nbsp;"
                    st.markdown(
                        f'<div style="background:linear-gradient(135deg,#ffffff 0%,#fce7f3 50%,#fbcfe8 100%);border-radius:10px;padding:14px 18px;margin-top:6px;font-size:.92rem;color:#831843;border:1px solid #ec4899;">'
                        f'{forecast_lines}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        with ai_tab3:
            st.caption("基于 K-Means 的天气模式聚类分析")
            cluster_result = cluster_weather(daily_df)
            if cluster_result is None:
                st.warning("scikit-learn 未安装或数据点不足，无法进行聚类分析。")
            else:
                st.metric("今日天气类型", cluster_result.current_cluster_name)
                st.caption(f"K-Means 聚类数: {cluster_result.n_clusters} | Inertia: {cluster_result.inertia:.2f}")
                if cluster_result.elbow_data:
                    elbow_df = pd.DataFrame(cluster_result.elbow_data, columns=["k", "Inertia"])
                    elbow_df = elbow_df.set_index("k")
                    st.line_chart(elbow_df, height=220)
                st.caption("聚类中心")
                centroid_rows = []
                for i, (c, name) in enumerate(zip(cluster_result.centroids, cluster_result.cluster_names)):
                    row = {"类型": name}
                    row.update(c)
                    centroid_rows.append(row)
                if centroid_rows:
                    st.dataframe(pd.DataFrame(centroid_rows), hide_index=True, use_container_width=True)

    except WeatherServiceError as exc:
        st.error(str(exc))

st.divider()
st.caption("天气数据：Open-Meteo · 预报存在不确定性，极端天气请以当地气象部门预警为准。")
