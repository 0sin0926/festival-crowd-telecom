"""평소 모델을 여러 방법으로 만들어 비교한다 (시계열 교차검증, 튜닝, 변수 제거 실험, 변수 중요도).

대상: 최근(2023.01~2026.07)의 행사 없는 토·일·공휴일. 구역 6개를 각각 예측한다.

비교하는 방법
  1. 요일유형×시간 평균            단순 평균
  2. 월×요일유형×시간 평균         단순 평균 (계절 반영)
  3. 선형회귀(Ridge)               요일유형×시각, 월을 범주로 넣고 날씨를 더함
  4. 랜덤포레스트
  5. 그래디언트 부스팅 (기본 설정)  지금까지 쓰던 모델
  6. 그래디언트 부스팅 (튜닝)       설정 후보 중 학습 구간 안의 검증으로 고름
변수 제거 실험 (그래디언트 부스팅 기본 설정에서)
  7. 날씨 변수 제거   8. 추세 변수 제거   9. 달력만 (날씨·추세 제거)

검증: 뒤쪽 6개월씩 4구간을 차례로 검증 구간으로 두고, 그 앞 전체로 학습한다(시간 순서를 지킴).
결과는 outputs/model_compare/ 에 저장한다.

사용법:
    python3 scripts/model_compare.py
"""
import itertools, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import baseline_model as b

warnings.filterwarnings("ignore")
OUT = b.ROOT / "outputs/model_compare"
ZONES = b.ZONES
FOLDS = [("2024-07-01", "2024-12-31"), ("2025-01-01", "2025-06-30"), ("2025-07-01", "2025-12-31"), ("2026-01-01", "2026-07-31")]
WEATHER = ["temperature_2m", "apparent_temperature", "precipitation", "precip_3h", "precip_day", "snowfall",
           "relative_humidity_2m", "wind_speed_10m", "cloud_cover"]
CALENDAR = [f for f in b.FEATURES if f not in WEATHER and f != "trend"]
GROUPS = {"시각": ["hour"], "요일·공휴일": ["dow", "holiday", "daytype", "pre_holiday"], "계절": ["month", "doy_sin", "doy_cos"],
          "추세": ["trend"], "기온": ["temperature_2m", "apparent_temperature"], "비·눈": ["precipitation", "precip_3h", "precip_day", "snowfall"],
          "습도": ["relative_humidity_2m"], "바람": ["wind_speed_10m"], "구름": ["cloud_cover"]}
BASE = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)   # 비교 기준이 되는 기본 설정
GRID = ([dict(learning_rate=lr, max_leaf_nodes=ml, min_samples_leaf=20, l2_regularization=1.0)
         for lr, ml in itertools.product([0.05, 0.1], [7, 15, 31])]
        + [dict(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0),
           dict(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=0.0)])   # 후보 8개 (한 번 학습에 수 초라 후보를 줄임)


def hgb(**kw):
    return HistGradientBoostingRegressor(**{**BASE, **kw})


def ridge():
    num = WEATHER + ["trend", "doy_sin", "doy_cos"]
    ct = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["dt_hour", "month", "pre_holiday"]),
                            ("num", StandardScaler(), num)])
    return make_pipeline(ct, Ridge(alpha=1.0))


def profile(train, test, z, keys):
    m = train.groupby(keys)[z].mean().rename("p")
    fb = train.groupby(["daytype", "hour"])[z].mean().rename("fb")
    o = test.join(m, on=keys).join(fb, on=["daytype", "hour"])
    return o["p"].fillna(o["fb"]).values


def mape(y, p):
    return float(np.mean(np.abs(y - p) / np.maximum(y, 1)) * 100)


def fit_predict(model, train, test, z, cols):
    model.fit(train[cols], np.log1p(train[z]))
    return np.expm1(model.predict(test[cols]))


def tune(train, z):
    """학습 구간의 마지막 20% 날짜를 떼어 후보 설정을 평가하고 가장 좋은 것을 돌려준다."""
    days = np.sort(train.date.unique()); cut = days[int(len(days) * 0.8)]
    tr, va = train[train.date < cut], train[train.date >= cut]
    scores = [mape(va[z].values, fit_predict(hgb(**g), tr, va, z, b.FEATURES)) for g in GRID]
    return GRID[int(np.argmin(scores))], scores


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    b.set_korean_font()
    d = b.load(); day = b.flag_events(d)
    n = d[d.date.isin(day.index[~day.event & ~day.incomplete]) & (d.daytype > 0)].copy()
    n["dt_hour"] = n.daytype.astype(str) + "_" + n.hour.astype(str)
    rows, chosen, grid_scores, imp = [], [], np.zeros(len(GRID)), []
    for k, (s, e) in enumerate(FOLDS, 1):
        train, test = n[n.date < s], n[(n.date >= s) & (n.date <= e)]
        print(f"구간 {k}: 학습 {train.date.nunique()}일 (~{train.date.max():%Y-%m-%d}), 검증 {test.date.nunique()}일 ({s}~{e})", flush=True)
        daytime = test.hour.between(10, 22).values
        for z in ZONES:
            print(f"  {z}", flush=True)
            best, sc = tune(train, z); grid_scores += np.array(sc); chosen.append({"구간": k, "구역": z, **best})
            preds = {
                "1. 요일유형×시간 평균": profile(train, test, z, ["daytype", "hour"]),
                "2. 월×요일유형×시간 평균": profile(train, test, z, ["month", "daytype", "hour"]),
                "3. 선형회귀(Ridge)": fit_predict(ridge(), train, test, z, ["dt_hour", "month", "pre_holiday"] + WEATHER + ["trend", "doy_sin", "doy_cos"]),
                "4. 랜덤포레스트": fit_predict(RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=0), train, test, z, b.FEATURES),
                "5. 그래디언트 부스팅 (기본)": fit_predict(hgb(), train, test, z, b.FEATURES),
                "6. 그래디언트 부스팅 (튜닝)": fit_predict(hgb(**best), train, test, z, b.FEATURES),
                "7. (실험) 날씨 변수 제거": fit_predict(hgb(), train, test, z, CALENDAR + ["trend"]),
                "8. (실험) 추세 변수 제거": fit_predict(hgb(), train, test, z, CALENDAR + WEATHER),
                "9. (실험) 달력만": fit_predict(hgb(), train, test, z, CALENDAR),
            }
            y = test[z].values
            for name, p in preds.items():
                rows.append({"구간": k, "구역": z, "방법": name, "MAPE": mape(y, p), "MAE": float(np.mean(np.abs(y - p))),
                             "낮 10~22시 MAPE": mape(y[daytime], p[daytime])})
            if k == len(FOLDS):                  # 변수 묶음별 중요도: 마지막 구간에서 그 묶음을 뒤섞었을 때 오차가 얼마나 느는지
                m = hgb().fit(train[b.FEATURES], np.log1p(train[z])); base = mape(y, np.expm1(m.predict(test[b.FEATURES])))
                rng = np.random.default_rng(0)
                for g, cols in GROUPS.items():
                    inc = []
                    for _ in range(5):
                        x = test[b.FEATURES].copy(); x[cols] = x[cols].values[rng.permutation(len(x))]
                        inc.append(mape(y, np.expm1(m.predict(x))) - base)
                    imp.append({"구역": z, "변수 묶음": g, "오차 증가(%p)": round(float(np.mean(inc)), 2)})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "구간별_결과.csv", index=False, encoding="utf-8-sig")
    avg = r.groupby(["구역", "방법"])[["MAPE", "MAE", "낮 10~22시 MAPE"]].mean().round(1).reset_index()
    avg.to_csv(OUT / "방법별_평균.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    t = avg.pivot(index="방법", columns="구역", values="MAPE")[ZONES]; t["6구역 평균"] = t.mean(axis=1).round(1)
    print("\n== 방법별 평균 오차율 MAPE(%) — 4구간 평균"); print(t.to_string())
    t2 = avg.pivot(index="방법", columns="구역", values="낮 10~22시 MAPE")[ZONES]; t2["6구역 평균"] = t2.mean(axis=1).round(1)
    print("\n== 낮 10~22시 MAPE(%)"); print(t2.to_string())
    t3 = avg.pivot(index="방법", columns="구역", values="MAE")[ZONES].round(0).astype(int)
    print("\n== MAE(명)"); print(t3.to_string())
    sd = r.groupby(["구역", "방법"])["MAPE"].agg(["min", "max"]).round(1).reset_index()
    print("\n== 그래디언트 부스팅 (기본)의 구간별 MAPE 범위"); print(sd[sd["방법"].str.startswith("5.")].drop(columns="방법").to_string(index=False))
    t.to_csv(OUT / "방법별_MAPE_표.csv", encoding="utf-8-sig")

    ch = pd.DataFrame(chosen); ch.to_csv(OUT / "튜닝_선택된_설정.csv", index=False, encoding="utf-8-sig")
    g = pd.DataFrame(GRID); g["학습구간 안 검증 MAPE 평균"] = (grid_scores / (len(FOLDS) * len(ZONES))).round(2)
    g = g.sort_values("학습구간 안 검증 MAPE 평균"); g.to_csv(OUT / "튜닝_후보별_점수.csv", index=False, encoding="utf-8-sig")
    print("\n== 튜닝 후보 상위 5개와 기본 설정"); print(g.head(5).to_string(index=False))
    print(g[(g.learning_rate == 0.05) & (g.max_leaf_nodes == 31) & (g.min_samples_leaf == 20) & (g.l2_regularization == 1.0)].to_string(index=False))

    im = pd.DataFrame(imp); im.to_csv(OUT / "변수_중요도.csv", index=False, encoding="utf-8-sig")
    ti = im.pivot(index="변수 묶음", columns="구역", values="오차 증가(%p)")[ZONES].loc[list(GROUPS)]
    print("\n== 변수 묶음을 뒤섞었을 때 오차율 증가(%p) — 마지막 구간"); print(ti.to_string())

    fig, ax = plt.subplots(figsize=(13, 6))
    show = t.drop(columns="6구역 평균").iloc[:6]
    show.T.plot.bar(ax=ax, width=0.85)
    ax.set_ylabel("오차율 MAPE(%)"); ax.set_title("평소 모델 비교 (시계열 교차검증 4구간 평균, 낮을수록 좋음)"); ax.legend(fontsize=8); plt.xticks(rotation=0)
    fig.tight_layout(); fig.savefig(OUT / "모델_비교.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ti.T.plot.bar(ax=ax, width=0.85); ax.set_ylabel("오차율 증가(%p)"); ax.set_title("변수 묶음별 중요도 (뒤섞었을 때 오차가 느는 정도)"); plt.xticks(rotation=0); ax.legend(fontsize=8, ncol=3)
    fig.tight_layout(); fig.savefig(OUT / "변수_중요도.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
