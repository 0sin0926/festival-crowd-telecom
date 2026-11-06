"""불꽃축제 날의 구역별·시간대별 인원을 예측한다.

    예측 인원 = 평소 인원 + 축제 증가분 × 보정 계수

- 평소 인원: 최근(2023.01~2026.07)의 행사 없는 토·일·공휴일로 학습한 모델에 그날의 요일·시각·날씨를 넣은 값
- 축제 증가분: 2022~2025년 불꽃축제 4회의 시간대별 증가분 평균
- 보정 계수: 맑은 날은 1. "오전 비 후 갬"은 2018년 증가분 ÷ 2017·2019년 평균 증가분 (시간대별)
두 시나리오(맑은 날 / 오전 비 후 갬)를 항상 함께 낸다. 결과는 outputs/predict/ 에 저장한다.

먼저 scripts/fireworks_increment.py 를 돌려 둬야 한다 (증가분 표를 읽어 쓴다).

사용법:
    python3 scripts/predict_fireworks.py 2026-09-05                       # 날씨는 그 시기 평년값
    python3 scripts/predict_fireworks.py 2026-09-05 --tmax 28 --tmin 20   # 최고·최저 기온을 지정
    python3 scripts/predict_fireworks.py 2026-09-05 --weather 예보.csv     # 시간별 날씨 파일(24행)을 직접 지정
    python3 scripts/predict_fireworks.py --backtest                       # 최근 4회를 하나씩 빼고 맞혀 보기
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
import baseline_model as b

OUT = b.ROOT / "outputs/predict"
INC = b.ROOT / "outputs/fireworks/불꽃축제_증가분_시간별.csv"
ZONES = b.ZONES
RECENT = ["2022-10-08", "2023-10-07", "2024-10-05", "2025-09-27"]     # 증가분 틀에 쓰는 회차
RAIN, RAIN_REF = "2018-10-06", ["2017-09-30", "2019-10-05"]            # 보정 계수: 비 온 해 ÷ 같은 시기 맑은 해
WVARS = ["temperature_2m", "apparent_temperature", "precipitation", "rain", "snowfall",
         "relative_humidity_2m", "wind_speed_10m", "cloud_cover"]
SCENARIOS = ["맑은 날", "오전 비 후 갬"]


def train_models():
    d = b.load()
    day = b.flag_events(d)
    n = d[d.date.isin(day.index[~day.event & ~day.incomplete]) & (d.daytype > 0)]
    return {z: b.make_model().fit(n[b.FEATURES], np.log1p(n[z])) for z in ZONES}


def templates(exclude=None):
    """증가분 틀(구역 × 시각)과 보정 계수(구역 × 시각)를 돌려준다. exclude: 틀에서 뺄 회차."""
    inc = pd.read_csv(INC)
    use = [t for t in RECENT if t != exclude]
    tmpl = inc[inc["날짜"].isin(use)].pivot_table(index="시각", columns="구역", values="증가분_모델", aggfunc="mean")[ZONES]
    past = inc.pivot_table(index=["구역", "시각"], columns="날짜", values="증가분_같은달")
    ref = past[RAIN_REF].mean(axis=1).unstack(0)[ZONES]
    k = (past[RAIN].unstack(0)[ZONES] / ref).clip(0, 1.2)
    for z in ZONES:                              # 증가분이 아주 작은 이른 시간대는 비율이 불안정 → 처음 안정된 시각의 값으로 채움
        ok = ref[z] >= 0.05 * ref[z].max()
        k.loc[~ok, z] = np.nan
        k[z] = k[z].bfill().ffill()
    return tmpl, k


def weather_for(date, tmax=None, tmin=None, path=None):
    """그날의 시간별 날씨(24행) 두 벌을 만든다: 맑은 날 / 오전 비 후 갬."""
    if path:
        w = pd.read_csv(path).iloc[:24][WVARS].reset_index(drop=True)
    else:                                        # 평년값: 받은 기간에서 같은 시기(앞뒤 10일)의 시각별 중앙값
        h = pd.read_csv(b.WEATHER, index_col=0, parse_dates=True)
        doy = h.index.dayofyear
        near = h[np.minimum(abs(doy - date.dayofyear), 365 - abs(doy - date.dayofyear)) <= 10]
        w = near.groupby(near.index.hour)[WVARS].median().reset_index(drop=True)
        if tmax is not None and tmin is not None:  # 하루 곡선 모양은 유지하고 최고·최저만 맞춤
            t = w["temperature_2m"]
            new = tmin + (t - t.min()) / (t.max() - t.min()) * (tmax - tmin)
            w["apparent_temperature"] += new - t
            w["temperature_2m"] = new
    clear = w.copy()
    clear[["precipitation", "rain", "snowfall"]] = 0.0
    rain = clear.copy()                          # 비·구름·습도는 2018-10-06의 실제 값(14시까지 비, 15시부터 갬)을 그대로 씀
    r = pd.read_csv(b.WEATHER_PAST, index_col=0, parse_dates=True).loc[RAIN]
    for v in ["precipitation", "rain", "cloud_cover", "relative_humidity_2m"]:
        rain[v] = r[v].values
    return {"맑은 날": clear, "오전 비 후 갬": rain}


def features(date, w):
    x = w.copy()
    x["hour"] = range(24)
    x["dow"] = date.dayofweek
    x["holiday"] = int(date in b.HOLIDAYS)
    x["daytype"] = 2 if (x["holiday"][0] or date.dayofweek == 6) else (1 if date.dayofweek == 5 else 0)
    nxt = date + pd.Timedelta(days=1)
    x["pre_holiday"] = int(nxt in b.HOLIDAYS or nxt.dayofweek >= 5)
    x["month"] = date.month
    x["doy_sin"] = np.sin(2 * np.pi * date.dayofyear / 365.25)
    x["doy_cos"] = np.cos(2 * np.pi * date.dayofyear / 365.25)
    x["trend"] = max((date - b.TREND_START).days, 0)
    x["precip_3h"] = x["precipitation"].rolling(3, min_periods=1).sum()
    x["precip_day"] = x["precipitation"].sum()
    return x[b.FEATURES]


def predict(date, models, tmpl, k, weathers):
    rows = []
    for sc in SCENARIOS:
        x = features(date, weathers[sc])
        for z in ZONES:
            base = np.expm1(models[z].predict(x))
            coef = k[z].values if sc == "오전 비 후 갬" else np.ones(24)
            inc = tmpl[z].values * coef
            rows.append(pd.DataFrame({"날짜": date.strftime("%Y-%m-%d"), "시나리오": sc, "구역": z, "시각": range(24),
                                      "평소": base.round(0), "증가분_틀": tmpl[z].values.round(0), "보정계수": np.round(coef, 2),
                                      "증가분": inc.round(0), "예측": (base + inc).round(0)}))
    return pd.concat(rows, ignore_index=True)


def summarize(p):
    out = []
    for (sc, z), g in p.groupby(["시나리오", "구역"], sort=False):
        g = g.set_index("시각"); i = g["예측"].idxmax()
        dbl = g.index[g["예측"] >= 2 * g["평소"]]
        out.append({"시나리오": sc, "구역": z, "피크 시각": i, "피크 예측": int(g.loc[i, "예측"]), "그때 평소": int(g.loc[i, "평소"]),
                    "그때 증가분": int(g.loc[i, "증가분"]), "평소의 2배 이상인 시간대": f"{dbl.min()}~{dbl.max()}시" if len(dbl) else "없음",
                    "시간 수": len(dbl)})
    return pd.DataFrame(out)


def plot(p, date, path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for ax, z in zip(axes.ravel(), ZONES):
        g = p[p["구역"] == z]
        c = g[g["시나리오"] == "맑은 날"]
        ax.fill_between(c["시각"], 0, c["평소"], color="lightgray", label="평소")
        ax.plot(c["시각"], c["예측"], label="맑은 날 예측")
        r = g[g["시나리오"] == "오전 비 후 갬"]
        ax.plot(r["시각"], r["예측"], ls="--", label="오전 비 후 갬 예측")
        ax.set_title(z); ax.set_xlabel("시각"); ax.set_ylabel("인원(명)"); ax.set_xticks(range(0, 24, 3))
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(f"불꽃축제 예측 {date:%Y-%m-%d} (평소 + 증가분 x 보정 계수)"); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def backtest(models):
    """최근 4회를 하나씩 빼고, 나머지 3회의 증가분 틀과 그날의 실제 날씨로 맞혀 본다."""
    inc = pd.read_csv(INC); rows = []
    for t in RECENT:
        tmpl, _ = templates(exclude=t)
        a = inc[inc["날짜"] == t]
        for z in ZONES:
            g = a[a["구역"] == z].set_index("시각")
            pred = g["평소_모델"] + tmpl[z]
            eve = slice(15, 22)
            rows.append({"날짜": t, "구역": z, "실제 피크": int(g["실제"].max()), "예측 피크": int(pred.max()),
                         "피크 오차(%)": round((pred.max() - g["실제"].max()) / g["실제"].max() * 100, 1),
                         "실제 피크 시각": int(g["실제"].idxmax()), "예측 피크 시각": int(pred.idxmax()),
                         "15~22시 평균 오차율(%)": round(((pred - g["실제"]).abs() / g["실제"]).loc[eve].mean() * 100, 1)})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "불꽃축제_하나씩빼고_검증.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    for z in ["여의동 전체", "공원 동쪽", "진입부", "동남쪽 끝", "업무지구", "서여의도"]:
        print(f"\n== {z}"); print(r[r["구역"] == z].drop(columns="구역").to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date", nargs="?")
    ap.add_argument("--tmax", type=float); ap.add_argument("--tmin", type=float)
    ap.add_argument("--weather"); ap.add_argument("--backtest", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    b.set_korean_font()
    models = train_models()
    if a.backtest:
        return backtest(models)
    date = pd.Timestamp(a.date)
    tmpl, k = templates()
    w = weather_for(date, a.tmax, a.tmin, a.weather)
    p = predict(date, models, tmpl, k, w)
    p.to_csv(OUT / f"불꽃축제_예측_{date:%Y-%m-%d}.csv", index=False, encoding="utf-8-sig")
    s = summarize(p)
    s.to_csv(OUT / f"불꽃축제_예측_{date:%Y-%m-%d}_요약.csv", index=False, encoding="utf-8-sig")
    plot(p, date, OUT / f"불꽃축제_예측_{date:%Y-%m-%d}.png")
    pd.set_option("display.width", 250)
    c = w["맑은 날"]
    print(f"{date:%Y-%m-%d} ({'월화수목금토일'[date.dayofweek]}) / 넣은 기온: 최저 {c.temperature_2m.min():.1f}도, 최고 {c.temperature_2m.max():.1f}도, 19시 {c.temperature_2m[19]:.1f}도")
    print(s.to_string(index=False))
    t = p[(p["구역"] == "여의동 전체") & p["시각"].between(12, 23)].pivot(index="시각", columns="시나리오", values=["평소", "증가분", "예측"])
    print("\n여의동 전체, 12~23시"); print(t.astype(int).to_string())


if __name__ == "__main__":
    main()
