# 逐风天气

一个使用 Python 和 Streamlit 开发的天气预报系统。用户输入地点后，可以查看：

- 当前天气、气温和体感温度
- 相对湿度、云量、风速和风向
- 当前紫外线指数及风险等级
- 未来 24 小时气温和降水概率
- 未来 3 天天气概览
- 晴转雨、降水开始、明显升降温和强风等短时变化提示

数据来自 [Open-Meteo](https://open-meteo.com/)，无需申请 API Key。

## 运行

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

浏览器通常会自动打开 `http://localhost:8501`。

## 测试

```powershell
python -m pytest
```

## 项目结构

```text
app.py                    Streamlit 页面和图表
weather_service.py        地点搜索、天气接口和数据整理
weather_analysis.py       短时天气变化判断
weather_codes.py          WMO 天气代码中文映射
tests/                    单元测试
```

## 变化预报规则

系统读取逐小时预报，并在未来 12 小时中识别：

1. 天气大类变化，如晴朗转为降雨或雷暴。
2. 降水概率达到 50%，或预报降水量达到 0.1 mm。
3. 未来 3 小时气温变化达到 3℃。
4. 最大阵风达到 50 km/h。

这些提示是对数值天气预报的规则化解读，不是自行训练的气象模型。极端天气请以当地气象部门发布的信息为准。
