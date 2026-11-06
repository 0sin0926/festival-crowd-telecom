"""서울시 문화행사 정보(OA-15486)에서 여의도 행사를 골라 행사 목록과 날짜별 달력을 만든다.
행사일은 인원수가 아니라 이 공식 목록으로만 정한다.

여의도 행사로 보는 기준 (둘 중 하나)
  - 좌표가 여의동 집계구 경계 안에 있다
  - 장소나 행사명에 여의도·여의나루·물빛광장 등이 들어 있다 (한강공원 여러 곳에서 함께 여는 행사 포함)

달력에서 "행사일"로 치는 것은 야외에서 열리고 기간이 LONG_DAYS일 이하인 행사만이다.
실내 공연장 행사와 몇 달씩 이어지는 상설 프로그램은 목록에는 남기되 행사일로 치지 않는다.
문화행사 목록에 빠져 있는 큰 행사는 data/raw/events/manual_events.csv 에 출처와 함께 직접 적어 넣는다.

사용법:
    python3 scripts/build_event_calendar.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/raw/events/seoul_cultural_events.csv"
MANUAL = ROOT / "data/raw/events/manual_events.csv"
GEO = ROOT / "data/processed/yeouido_jipgyegu.geojson"
PANEL = ROOT / "data/processed/yeouido_zone_hourly.csv"
OUT_EVENTS = ROOT / "data/processed/yeouido_events.csv"
OUT_CAL = ROOT / "data/processed/yeouido_event_calendar.csv"
START, END = "2023-01-01", "2026-07-31"      # 생활인구를 받는 기간
LONG_DAYS = 14
PLACE_KW = "여의도|여의나루|물빛광장|여의별빛|선유-여의"
TITLE_KW = "여의도|불꽃축제"
INDOOR_KW = "홀|회의실|전시실|강당|벙커|로비|볼룸|빌딩|센터|도서관|체험교실|수영장"      # 이 말이 장소에 있으면 실내로 본다
RIVER = ["공원 동쪽", "진입부", "동남쪽 끝"]

from baseline_model import HOLIDAYS      # 공휴일 목록은 baseline_model.py 한 곳에서 관리


def in_yeouido(lon, lat, polys):
    def inside(ring):
        hit, n = False, len(ring)
        for i in range(n):
            (x1, y1), (x2, y2) = ring[i], ring[(i + 1) % n]
            if (y1 > lat) != (y2 > lat) and lon < (x2 - x1) * (lat - y1) / (y2 - y1) + x1:
                hit = not hit
        return hit
    return any(inside(r) for r in polys)


def load_polys():
    polys = []
    for f in json.loads(GEO.read_text(encoding="utf-8"))["features"]:
        g = f["geometry"]
        for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
            polys.append(poly[0])
    return polys


def main():
    d = pd.read_csv(SRC, encoding="cp949", encoding_errors="replace", dtype=str).fillna("")
    d.columns = [c.replace("?", " ") for c in d.columns]
    a = pd.to_numeric(d["경도(X좌표)"], errors="coerce"); b = pd.to_numeric(d["위도(Y좌표)"], errors="coerce")
    lon, lat = np.where(a > 100, a, b), np.where(a > 100, b, a)      # 경도·위도가 뒤바뀐 행이 있어 큰 쪽을 경도로
    polys = load_polys()
    by_xy = np.array([not np.isnan(x) and in_yeouido(x, y, polys) for x, y in zip(lon, lat)])
    by_kw = d["장소"].str.contains(PLACE_KW) | d["공연/행사명"].str.contains(TITLE_KW)
    e = d[by_xy | by_kw].copy()
    e["시작일"] = pd.to_datetime(e["시작일"].str[:10]); e["종료일"] = pd.to_datetime(e["종료일"].str[:10])
    e = e[(e["종료일"] >= START) & (e["시작일"] <= END)]
    e["기간_일"] = (e["종료일"] - e["시작일"]).dt.days + 1
    e["판정근거"] = np.where(by_xy[e.index] & by_kw[e.index], "좌표+이름", np.where(by_xy[e.index], "좌표", "이름"))
    e["출처"] = "서울시 문화행사 정보"
    m = pd.read_csv(MANUAL, dtype=str).fillna("")
    m["시작일"] = pd.to_datetime(m["시작일"]); m["종료일"] = pd.to_datetime(m["종료일"])
    m["기간_일"] = (m["종료일"] - m["시작일"]).dt.days + 1
    m["판정근거"] = "직접 추가"
    e = pd.concat([e, m], ignore_index=True).fillna("")
    e["야외"] = ~e["장소"].str.contains(INDOOR_KW)
    e["행사일_반영"] = e["야외"] & (e["기간_일"] <= LONG_DAYS)
    cols = ["시작일", "종료일", "기간_일", "공연/행사명", "분류", "장소", "행사시간", "야외", "행사일_반영", "판정근거",
            "자치구", "기관명", "문화포털상세URL", "출처"]
    e = e[cols].sort_values(["시작일", "종료일"]).drop_duplicates(["시작일", "종료일", "공연/행사명", "장소"])
    e.to_csv(OUT_EVENTS, index=False, encoding="utf-8-sig")

    cal = pd.DataFrame(index=pd.date_range(START, END, name="date"))
    cal["요일"] = cal.index.dayofweek.map(dict(enumerate("월화수목금토일")))
    hol = cal.index.isin(HOLIDAYS)
    cal["요일유형"] = np.where(hol | (cal.index.dayofweek == 6), "일·공휴일", np.where(cal.index.dayofweek == 5, "토", "평일"))
    names, others = {t: [] for t in cal.index}, {t: 0 for t in cal.index}
    for r in e.itertuples():
        for t in pd.date_range(max(r.시작일, pd.Timestamp(START)), min(r.종료일, pd.Timestamp(END))):
            if r.행사일_반영:
                names[t].append(r._4)
            else:
                others[t] += 1
    cal["행사일"] = [len(names[t]) > 0 for t in cal.index]
    cal["행사_수"] = [len(names[t]) for t in cal.index]
    cal["행사명"] = [" / ".join(names[t]) for t in cal.index]
    cal["그외_행사_수(실내·장기)"] = [others[t] for t in cal.index]
    if PANEL.exists():                           # 참고용: 받은 달의 강변 구역(공원 동쪽+진입부+동남쪽 끝) 하루 최대 인원
        p = pd.read_csv(PANEL, index_col=0, parse_dates=True)
        cal["강변_최대인원(받은 달만)"] = p[RIVER].sum(axis=1).groupby(p.index.normalize()).max().round(0)
    cal.to_csv(OUT_CAL, encoding="utf-8-sig")

    print(f"여의도 행사 {len(e)}건 (그중 행사일로 반영 {int(e.행사일_반영.sum())}건)")
    print(pd.crosstab(cal["요일유형"], cal["행사일"]).rename(columns={False: "행사 없는 날", True: "행사일"}).to_string())


if __name__ == "__main__":
    main()
