"""평소 곡선(축제가 없는 날의 구역별·시간대별 인원) 모델.

1. 구역 × 시간 표에 달력(요일·공휴일·계절)과 날씨를 붙인다
2. 공식 행사 달력에 있는 날을 빼 평소 표본을 정한다
3. 구역별로 모델을 학습하고, 뒤쪽 기간으로 검증한다 (단순 평균 방식과 비교)
4. 결과 표와 그림을 outputs/baseline/ 에 저장한다

먼저 scripts/build_panel.py 와 scripts/build_event_calendar.py 를 돌려 둬야 한다.

사용법:
    python3 scripts/baseline_model.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "data/processed/yeouido_zone_hourly.csv"
WEATHER = ROOT / "data/processed/weather_yeouido_hourly.csv"
WEATHER_PAST = ROOT / "data/processed/weather_yeouido_hourly_past.csv"   # 과거 불꽃축제 달(2017·2018·2019·2022) 전후
TREND_START = pd.Timestamp("2023-01-01")     # 추세 변수의 기준일 (학습 기간의 첫날)
CALENDAR = ROOT / "data/processed/yeouido_event_calendar.csv"
OUT = ROOT / "outputs/baseline"
ZONES = ["공원 동쪽", "진입부", "서여의도", "동남쪽 끝", "업무지구", "여의동 전체"]
PARK = ["공원 동쪽", "진입부", "서여의도", "동남쪽 끝"]
WEEKEND_ONLY = True          # True면 토·일·공휴일만 학습·검증한다 (축제가 열리는 날과 같은 조건). False면 평일 포함

# 공휴일(대체·임시 공휴일, 선거일 포함). 2024~2026년은 2026-10-11에 검색으로 대조함(2026-07-17 제헌절 추가).
# 2023년과 과거 달(2017~2022)은 기억 기준. 5월 1일(근로자의 날)은 관공서 공휴일은 아니지만
# 여의도는 금융·업무지구라 대부분 쉬므로 공휴일로 넣었다.
HOLIDAYS = pd.to_datetime([
    "2017-10-02", "2017-10-03", "2017-10-04", "2017-10-05", "2017-10-06", "2017-10-09",
    "2018-09-24", "2018-09-25", "2018-09-26", "2018-10-03", "2018-10-09",
    "2019-09-12", "2019-09-13", "2019-10-03", "2019-10-09",
    "2022-09-09", "2022-09-12", "2022-10-03", "2022-10-09", "2022-10-10",
    "2023-01-01", "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24", "2023-03-01", "2023-05-05",
    "2023-05-27", "2023-05-29", "2023-06-06", "2023-08-15", "2023-09-28", "2023-09-29", "2023-09-30",
    "2023-10-02", "2023-10-03", "2023-10-09", "2023-12-25",
    "2024-01-01", "2024-02-09", "2024-02-10", "2024-02-11", "2024-02-12", "2024-03-01", "2024-04-10",
    "2024-05-05", "2024-05-06", "2024-05-15", "2024-06-06", "2024-08-15", "2024-09-16", "2024-09-17",
    "2024-09-18", "2024-10-01", "2024-10-03", "2024-10-09", "2024-12-25",
    "2025-01-01", "2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30", "2025-03-01", "2025-03-03",
    "2025-05-05", "2025-05-06", "2025-06-03", "2025-06-06", "2025-08-15", "2025-10-03", "2025-10-05",
    "2025-10-06", "2025-10-07", "2025-10-08", "2025-10-09", "2025-12-25",
    "2026-01-01", "2026-02-16", "2026-02-17", "2026-02-18", "2026-03-01", "2026-03-02", "2026-05-05",
    "2026-05-24", "2026-05-25", "2026-06-03", "2026-06-06", "2026-07-17",
    "2023-05-01", "2024-05-01", "2025-05-01", "2026-05-01",
])
FEATURES = ["hour", "dow", "holiday", "daytype", "pre_holiday", "month", "doy_sin", "doy_cos", "trend",
            "temperature_2m", "apparent_temperature", "precipitation", "precip_3h", "precip_day",
            "snowfall", "relative_humidity_2m", "wind_speed_10m", "cloud_cover"]


# 평소 모델의 설정. scripts/model_compare.py 의 비교·튜닝 결과로 정한 값 (docs/작업기록.md 5-6)
MODEL_PARAMS = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)


def set_korean_font():
    """그림에 쓸 한글 글꼴을 운영체제에 있는 것으로 고른다 (맥, 윈도우, 리눅스)."""
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"]:
        if name in have:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def make_model(**override):
    return HistGradientBoostingRegressor(**{**MODEL_PARAMS, **override})


def load(past=False):
    """past=False면 학습 기간(2023.01~)만, True면 과거 불꽃축제 달까지 포함해 돌려준다."""
    p = pd.read_csv(PANEL, index_col=0, parse_dates=True)
    p = p[p["여의동 전체"] > 0].dropna()             # 값이 0이거나 비어 있는 시간은 누락으로 본다
    w = pd.read_csv(WEATHER, index_col=0, parse_dates=True)
    if past:
        w = pd.concat([pd.read_csv(WEATHER_PAST, index_col=0, parse_dates=True), w]).sort_index()
    w["precip_3h"] = w["precipitation"].rolling(3, min_periods=1).sum()
    w["precip_day"] = w.groupby(w.index.normalize())["precipitation"].transform("sum")
    d = p.join(w, how="inner")
    date = d.index.normalize()
    d["date"] = date
    d["hour"] = d.index.hour
    d["dow"] = d.index.dayofweek
    d["holiday"] = date.isin(HOLIDAYS).astype(int)
    d["daytype"] = np.where(d["holiday"].eq(1) | d["dow"].eq(6), 2, np.where(d["dow"].eq(5), 1, 0))  # 0 평일, 1 토, 2 일·공휴일
    nxt = date + pd.Timedelta(days=1)
    d["pre_holiday"] = (nxt.isin(HOLIDAYS) | (nxt.dayofweek >= 5)).astype(int)
    d["month"] = d.index.month
    d["doy_sin"] = np.sin(2 * np.pi * d.index.dayofyear / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * d.index.dayofyear / 365.25)
    d["trend"] = np.maximum((date - TREND_START).days, 0)
    return d


def flag_events(d):
    """행사일은 공식 행사 달력(build_event_calendar.py가 만든 파일)으로만 정한다."""
    cal = pd.read_csv(CALENDAR, index_col=0, parse_dates=True)
    day = d.groupby("date").agg(daytype=("daytype", "first"), hours=("hour", "size"))
    day["event"] = cal["행사일"].reindex(day.index).fillna(False).astype(bool)
    day["행사명"] = cal["행사명"].reindex(day.index).fillna("")
    day["incomplete"] = day.hours < 24
    return day


def split(days):
    """달이 30개 이상이면 마지막 12개월을, 아니면 마지막 20% 날짜를 검증용으로 둔다."""
    days = pd.DatetimeIndex(sorted(days))
    if days.to_period("M").nunique() >= 30:
        cut = (days.max().to_period("M") - 11).to_timestamp()
    else:
        cut = days[int(len(days) * 0.8)]
    return cut


def profile_predict(train, test, zone, keys):
    """단순 평균 방식: keys 조합별 학습 기간 평균. 조합이 없으면 (요일 유형, 시간) 평균으로 대신한다."""
    m = train.groupby(keys)[zone].mean().rename("pred")
    fallback = train.groupby(["daytype", "hour"])[zone].mean().rename("fb")
    out = test.join(m, on=keys).join(fallback, on=["daytype", "hour"])
    return out["pred"].fillna(out["fb"]).values


def score(y, pred, mask=None):
    y, pred = np.asarray(y), np.asarray(pred)
    if mask is not None:
        y, pred = y[mask], pred[mask]
    return np.mean(np.abs(y - pred)), np.mean(np.abs(y - pred) / np.maximum(y, 1)) * 100


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    set_korean_font()
    d = load()
    day = flag_events(d)
    day[day.event | day.incomplete].to_csv(OUT / "제외한_날.csv", encoding="utf-8-sig")
    normal_days = day.index[~day.event & ~day.incomplete]
    n = d[d.date.isin(normal_days)]
    if WEEKEND_ONLY:
        n = n[n.daytype > 0]
    cut = split(n.date.unique())
    train, test = n[n.date < cut], n[n.date >= cut]
    print(f"전체 {day.shape[0]}일 / 행사로 제외 {int(day.event.sum())}일 / 시간 누락으로 제외 {int(day.incomplete.sum())}일")
    print(f"학습 {train.date.nunique()}일 ({train.date.min():%Y-%m-%d}~{train.date.max():%Y-%m-%d}), "
          f"검증 {test.date.nunique()}일 ({test.date.min():%Y-%m-%d}~{test.date.max():%Y-%m-%d})")

    rows, preds, models = [], pd.DataFrame(index=test.index), {}
    weekend_day = (test.daytype.values > 0) & (test.hour.values >= 10) & (test.hour.values <= 22)
    for z in ZONES:
        model = make_model()
        model.fit(train[FEATURES], np.log1p(train[z]))
        models[z] = model
        cand = {
            "요일유형×시간 평균": profile_predict(train, test, z, ["daytype", "hour"]),
            "월×요일유형×시간 평균": profile_predict(train, test, z, ["month", "daytype", "hour"]),
            "모델(달력+날씨)": np.expm1(model.predict(test[FEATURES])),
        }
        preds[z] = cand["모델(달력+날씨)"]
        for name, pr in cand.items():
            mae, mape = score(test[z], pr)
            mae_w, mape_w = score(test[z], pr, weekend_day)
            rows.append({"구역": z, "방법": name, "MAE(명)": round(mae), "MAPE(%)": round(mape, 1),
                         "주말 10~22시 MAE(명)": round(mae_w), "주말 10~22시 MAPE(%)": round(mape_w, 1)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "검증_결과.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))

    # 그림 1: 요일 유형별 평균 곡선
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, z in zip(axes.ravel(), ZONES):
        for t, name in enumerate(["평일", "토요일", "일·공휴일"]):
            if (n.daytype == t).any():
                ax.plot(n[n.daytype == t].groupby("hour")[z].mean(), label=name, color=f"C{t}")
        ax.set_title(z); ax.set_xlabel("시각"); ax.set_ylabel("인원(명)"); ax.set_xticks(range(0, 24, 3)); ax.legend()
    fig.suptitle("평소 요일 유형별 평균 곡선 (행사일 제외)"); fig.tight_layout(); fig.savefig(OUT / "1_요일유형별_평균곡선.png", dpi=130); plt.close(fig)

    # 그림 2: 주말 15시, 기온·비에 따른 공원 동쪽 인원
    s = n[(n.daytype > 0) & (n.hour == 15)]
    wet = s.precip_day >= 1
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(s[~wet].temperature_2m, s[~wet]["공원 동쪽"], label="비 안 온 날", alpha=0.7)
    ax.scatter(s[wet].temperature_2m, s[wet]["공원 동쪽"], label="비 온 날 (하루 1mm 이상)", alpha=0.7, marker="x")
    ax.set_xlabel("15시 기온(°C)"); ax.set_ylabel("공원 동쪽 인원(명)"); ax.set_title("주말·공휴일 15시 — 기온·비와 공원 동쪽 인원"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "2_날씨와_공원동쪽.png", dpi=130); plt.close(fig)

    # 그림 3: 검증 기간 마지막 2주의 실제 vs 예측
    last = test[test.index >= test.index.max() - pd.Timedelta(days=14)]
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True)
    for ax, z in zip(axes, ["공원 동쪽", "여의동 전체"]):
        ax.plot(last.index, last[z], label="실제"); ax.plot(last.index, preds.loc[last.index, z], label="모델 예측", ls="--")
        ax.set_title(z); ax.set_ylabel("인원(명)"); ax.legend()
    fig.suptitle("검증 기간 마지막 2주 — 실제 vs 예측"); fig.tight_layout(); fig.savefig(OUT / "3_실제_vs_예측.png", dpi=130); plt.close(fig)

    # 표: 주말 낮(12~18시) 날씨 조건별 공원 구역 평균 인원
    a = n[(n.daytype > 0) & n.hour.between(12, 18)].copy()
    a["기온대"] = pd.cut(a.temperature_2m, [-30, 5, 15, 25, 45], labels=["5도 미만", "5~15도", "15~25도", "25도 이상"])
    a["비"] = np.where(a.precip_day >= 1, "비 온 날", "비 안 온 날")
    tab = a.groupby(["기온대", "비"], observed=True)[PARK].mean().round(0)
    tab["날 수"] = a.groupby(["기온대", "비"], observed=True).date.nunique()
    tab.to_csv(OUT / "날씨별_주말낮_평균.csv", encoding="utf-8-sig")
    print(tab.to_string())


if __name__ == "__main__":
    main()
