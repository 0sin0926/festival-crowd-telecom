"""불꽃축제 7회(2017·2018·2019·2022·2023·2024·2025)의 증가분을 구한다.

1. 평소 모델은 최근(2023.01~2026.07)의 행사 없는 토·일·공휴일로만 학습한다
2. 과거 축제 달(2017.09, 2018.10, 2019.10, 2022.10)의 다른 토·일·공휴일에 이 모델을 적용해
   실제 인원과 비교한다 → 최근 모델이 과거에도 통하는지 확인
3. 불꽃축제일의 증가분 = 실제 − 평소. 평소는 두 가지로 잡아 나란히 둔다
   - 모델: 그날의 요일·시각·날씨를 평소 모델에 넣은 값
   - 같은 달: 같은 달의 다른 토요일(행사 없는 날) 같은 시각 중앙값
결과는 outputs/fireworks/ 에 저장한다.

먼저 scripts/build_panel.py 와 scripts/build_event_calendar.py 를 돌려 둬야 한다.

사용법:
    python3 scripts/fireworks_increment.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
import baseline_model as b

OUT = b.ROOT / "outputs/fireworks"
ZONES = b.ZONES
FIREWORKS = ["2017-09-30", "2018-10-06", "2019-10-05", "2022-10-08", "2023-10-07", "2024-10-05", "2025-09-27"]
PAST_MONTHS = ["2017-09", "2018-10", "2019-10", "2022-10"]
# 과거 달의 평소 비교에서 빼는 날: 불꽃축제일과, 공식 목록·보도로 확인된 여의도 야외 행사일
PAST_EVENT_DAYS = pd.to_datetime(["2017-09-30", "2018-10-06", "2019-10-05", "2022-10-08"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    b.set_korean_font()
    d = b.load(past=True)
    day = b.flag_events(d)
    fw = pd.to_datetime(FIREWORKS)
    recent = d[d.date >= b.TREND_START]
    train = recent[recent.date.isin(day.index[~day.event & ~day.incomplete]) & (recent.daytype > 0)]
    models = {}
    for z in ZONES:
        models[z] = b.make_model().fit(train[b.FEATURES], np.log1p(train[z]))
    pred = pd.DataFrame({z: np.expm1(models[z].predict(d[b.FEATURES])) for z in ZONES}, index=d.index)

    # 1) 최근 모델이 과거 달의 평소(토·일·공휴일)에 통하는지
    rows = []
    for ym in PAST_MONTHS:
        m = d[(d.index.strftime("%Y-%m") == ym) & (d.daytype > 0) & ~d.date.isin(PAST_EVENT_DAYS)]
        for z in ZONES:
            err = (pred.loc[m.index, z] - m[z]) / m[z].clip(lower=1) * 100
            rows.append({"달": ym, "구역": z, "비교한 날 수": m.date.nunique(), "오차율 MAPE(%)": round(err.abs().mean(), 1),
                         "치우침(%)": round(err.mean(), 1),
                         "낮 10~22시 치우침(%)": round(err[m.hour.between(10, 22)].mean(), 1)})
    val = pd.DataFrame(rows)
    val.to_csv(OUT / "과거달_평소_검증.csv", index=False, encoding="utf-8-sig")
    print("== 최근 모델을 과거 달의 평소 토·일·공휴일에 적용 (치우침 +는 모델이 실제보다 높게 예측)")
    print(val.pivot(index="구역", columns="달", values="치우침(%)").loc[ZONES].to_string())
    print(val.pivot(index="구역", columns="달", values="오차율 MAPE(%)").loc[ZONES].to_string())
    print("비교한 날 수:", val.groupby("달")["비교한 날 수"].first().to_dict())

    # 2) 불꽃축제 7회의 증가분
    normal_days = set(day.index[~day.event]) - set(PAST_EVENT_DAYS)
    parts = []
    for t in fw:
        x = d[d.date == t]
        if x.empty:
            print(f"{t:%Y-%m-%d} 데이터 없음"); continue
        same = d[(d.index.strftime("%Y-%m") == t.strftime("%Y-%m")) | (abs((d.date - t).dt.days) <= 35)]
        same = same[(same.dow == 5) & (same.holiday == 0) & same.date.isin(normal_days) & (same.date != t)]
        for z in ZONES:
            base_same = same.groupby("hour")[z].median().reindex(x.hour.values).values
            parts.append(pd.DataFrame({"날짜": t.strftime("%Y-%m-%d"), "시각": x.hour.values, "구역": z, "실제": x[z].values.round(0),
                                       "평소_모델": pred.loc[x.index, z].values.round(0), "평소_같은달": np.round(base_same, 0),
                                       "같은달_토요일_수": same.date.nunique()}))
    inc = pd.concat(parts, ignore_index=True)
    inc["증가분_모델"] = inc["실제"] - inc["평소_모델"]
    inc["증가분_같은달"] = inc["실제"] - inc["평소_같은달"]
    inc.to_csv(OUT / "불꽃축제_증가분_시간별.csv", index=False, encoding="utf-8-sig")

    summ = []
    for (t, z), g in inc.groupby(["날짜", "구역"]):
        k = g["증가분_모델"].idxmax()
        w = d[(d.date == t) & (d.hour == 19)].iloc[0]
        summ.append({"날짜": t, "구역": z, "피크 시각": int(g.loc[k, "시각"]), "피크 실제": int(g.loc[k, "실제"]),
                     "피크 증가분(모델)": int(g.loc[k, "증가분_모델"]), "피크 증가분(같은달)": int(g.loc[k, "증가분_같은달"]),
                     "같은달 토요일 수": int(g["같은달_토요일_수"].iloc[0]),
                     "19시 기온": w.temperature_2m, "하루 강수(mm)": round(w.precip_day, 1)})
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "불꽃축제_요약.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    for z in ["여의동 전체", "공원 동쪽", "업무지구"]:
        print(f"\n== {z}"); print(summ[summ["구역"] == z].drop(columns="구역").to_string(index=False))
    tot = inc[inc["구역"] == "여의동 전체"].pivot(index="시각", columns="날짜", values="증가분_모델")
    print("\n== 여의동 전체 증가분 곡선끼리의 상관계수"); print(tot.corr().round(3).to_string())

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for ax, z in zip(axes.ravel(), ZONES):
        c = inc[inc["구역"] == z].pivot(index="시각", columns="날짜", values="증가분_모델")
        for col in c.columns:
            ax.plot(c.index, c[col], label=col, ls="--" if col < "2023" else "-")
        ax.set_title(z); ax.set_xlabel("시각"); ax.set_ylabel("증가분(명)"); ax.set_xticks(range(0, 24, 3)); ax.axhline(0, color="gray", lw=0.5)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("불꽃축제 회차별 증가분 (실제 - 평소 모델). 점선은 과거 회차"); fig.tight_layout()
    fig.savefig(OUT / "불꽃축제_증가분_회차별.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
