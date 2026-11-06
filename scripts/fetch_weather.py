"""여의도 지점의 시간별 과거 날씨를 Open-Meteo(ERA5 재분석 기반, 인증키 불필요)에서 받아 CSV로 저장한다.
기상청 ASOS 관측값이 아니라 격자 재분석값이다. 표준 라이브러리만 사용.

사용법:
    python3 scripts/fetch_weather.py 2023-01-01 2026-07-31
"""
import csv, json, sys, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/processed/weather_yeouido_hourly.csv"
LAT, LON = 37.5265, 126.9340          # 여의도한강공원 부근
VARS = ["temperature_2m", "apparent_temperature", "precipitation", "rain", "snowfall",
        "relative_humidity_2m", "wind_speed_10m", "cloud_cover", "weather_code"]

def main():
    start, end = sys.argv[1], sys.argv[2]
    q = urllib.parse.urlencode({"latitude": LAT, "longitude": LON, "start_date": start, "end_date": end,
                                "hourly": ",".join(VARS), "timezone": "Asia/Seoul"})
    with urllib.request.urlopen(f"https://archive-api.open-meteo.com/v1/archive?{q}", timeout=120) as r:
        h = json.load(r)["hourly"]
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time"] + VARS)
        w.writerows(zip(h["time"], *(h[v] for v in VARS)))
    print(f"{OUT.name}: {len(h['time'])}행 ({h['time'][0]} ~ {h['time'][-1]})")

if __name__ == "__main__":
    main()
