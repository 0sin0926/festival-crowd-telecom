"""월별 여의동 추출본(yeouido_YYYYMM.csv.gz)을 모아 구역 × 1시간 인원 표를 만든다.
구역은 큰 집계구 5개 + 그 외 + 여의동 전체. 받은 달이 늘어나면 다시 돌리면 된다.

사용법:
    python3 scripts/build_panel.py
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/processed/yeouido_pop"
OUT = ROOT / "data/processed/yeouido_zone_hourly.csv"
ZONES = {                     # 집계구코드 끝 6자리 -> 구역 이름 (docs/작업기록.md 4장)
    "030001": "공원 동쪽",
    "030101": "진입부",
    "010001": "서여의도",
    "060003": "동남쪽 끝",
    "030002": "업무지구",
}
ORDER = list(ZONES.values()) + ["그 외", "여의동 전체"]

def main():
    parts = []
    for f in sorted(SRC.glob("yeouido_*.csv.gz")):
        d = pd.read_csv(f, dtype=str, usecols=["기준일ID", "시간대구분", "집계구코드", "총생활인구수"])
        d["pop"] = pd.to_numeric(d["총생활인구수"], errors="coerce").fillna(0)   # '*'(3명 이하)는 0으로
        # 일부 날짜는 코드 뒤에 공백이 붙거나 시각이 '0'처럼 한 자리로 들어 있어 맞춰 준다
        d["zone"] = d["집계구코드"].str.strip().str[-6:].map(ZONES).fillna("그 외")
        d["time"] = pd.to_datetime(d["기준일ID"].str.strip() + d["시간대구분"].str.strip().str.zfill(2), format="%Y%m%d%H")
        parts.append(d.groupby(["time", "zone"], as_index=False)["pop"].sum())
    p = pd.concat(parts)
    total = p.groupby("time", as_index=False)["pop"].sum().assign(zone="여의동 전체")
    p = pd.concat([p, total]).pivot(index="time", columns="zone", values="pop")[ORDER].round(1).sort_index()
    p.to_csv(OUT, encoding="utf-8")
    days = p.index.normalize().nunique()
    months = sorted(set(p.index.strftime("%Y%m")))
    print(f"{OUT.name}: {len(p)}시간, {days}일, {len(months)}개월 ({months[0]} ~ {months[-1]})")
    full = pd.date_range(p.index.min(), p.index.max(), freq="h")
    print(f"빠진 시간: {len(full.difference(p.index))}개 (아직 안 받은 달 포함)")

if __name__ == "__main__":
    main()
