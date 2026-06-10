"""
EVMS (Earned Value Management System) 분석 모듈
PV / EV / AC → SV / CV / SPI / CPI / 예측값(EAC, ETC, VAC) 산출
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 원본 DB 재현 (material_management_db.py 와 동일)
# ============================================================

plan_data = {
    "wbs_code":       ["A-01", "A-01", "A-02", "A-02", "B-01", "B-01", "B-02"],
    "work_name":      ["기초공사", "기초공사", "골조공사", "골조공사", "지하1층 골조", "지하1층 골조", "지상1층 골조"],
    "material_name":  ["철근", "레미콘", "철근", "레미콘", "철골", "볼트", "철근"],
    "planned_date":   ["2024-03-01", "2024-03-01", "2024-04-01", "2024-04-01",
                       "2024-04-15", "2024-04-15", "2024-05-01"],
    "planned_qty":    [5000, 120, 8000, 200, 3000, 500, 6000],
    "unit":           ["kg", "m³", "kg", "m³", "kg", "EA", "kg"],
    "std_unit_price": [1050, 85000, 1050, 85000, 2200, 800, 1050],
    "location":       ["B1-기초", "B1-기초", "B1-골조", "B1-골조", "B1-전체", "B1-전체", "1F-골조"],
}
plan_df = pd.DataFrame(plan_data)
plan_df["planned_date"] = pd.to_datetime(plan_df["planned_date"])

BBOX_TO_QTY = {"철근": 50, "레미콘": 6, "철골": 200, "볼트": 10}

vision_data = {
    "capture_date":   ["2024-03-05", "2024-03-05", "2024-04-03", "2024-04-03",
                       "2024-04-18", "2024-04-18", "2024-05-02"],
    "location":       ["B1-기초", "B1-기초", "B1-골조", "B1-골조",
                       "B1-전체", "B1-전체", "1F-골조"],
    "material_name":  ["철근", "레미콘", "철근", "레미콘", "철골", "볼트", "철근"],
    "bbox_count":     [92, 18, 148, 31, 14, 46, 110],
    "confidence_avg": [0.91, 0.88, 0.93, 0.85, 0.90, 0.87, 0.92],
    "image_count":    [12, 8, 20, 10, 15, 15, 18],
}
vision_df = pd.DataFrame(vision_data)
vision_df["capture_date"] = pd.to_datetime(vision_df["capture_date"])
vision_df["actual_qty"] = vision_df["material_name"].map(BBOX_TO_QTY) * vision_df["bbox_count"]

cost_data = {
    "price_date":        ["2024-03-01", "2024-04-01", "2024-04-01", "2024-04-15",
                          "2024-05-01", "2024-03-01", "2024-04-15"],
    "material_name":     ["철근", "철근", "레미콘", "철골", "철근", "레미콘", "볼트"],
    "actual_unit_price": [1080, 1100, 87000, 2350, 1090, 86000, 820],
}
cost_df = pd.DataFrame(cost_data)
cost_df["price_date"] = pd.to_datetime(cost_df["price_date"])


# ============================================================
# Step 1. DB 병합 (날짜 ±7일, 자재명 기준)
# ============================================================

def build_base_df(plan_df, vision_df, cost_df, tolerance_days=7) -> pd.DataFrame:
    tol = pd.Timedelta(f"{tolerance_days}D")

    # Vision + Cost 병합
    vc = pd.merge_asof(
        vision_df.sort_values("capture_date"),
        cost_df.sort_values("price_date").rename(columns={"price_date": "capture_date"}),
        on="capture_date", by="material_name",
        direction="nearest", tolerance=tol,
    )

    # + Plan 병합
    base = pd.merge_asof(
        vc.sort_values("capture_date"),
        plan_df[["planned_date", "wbs_code", "work_name",
                 "material_name", "planned_qty", "std_unit_price", "unit",
                 "location"]].sort_values("planned_date"),
        left_on="capture_date", right_on="planned_date",
        by="material_name", direction="nearest", tolerance=tol,
    )
    return base


# ============================================================
# Step 2. EVMS 핵심 지표 계산
# ============================================================

def compute_evms(df: pd.DataFrame) -> pd.DataFrame:
    """
    입력 DataFrame에 EVMS 지표 컬럼을 추가하여 반환.

    지표 정의
    ---------
    PV  (Planned Value)   = 계획수량 × 기준단가          → 이 기간에 쓰기로 한 예산
    EV  (Earned Value)    = 실적수량 × 기준단가          → 실제로 완성한 작업의 가치
    AC  (Actual Cost)     = 실적수량 × 실제구매단가      → 실제로 지출한 비용

    SV  (Schedule Variance) = EV − PV   > 0: 공정 앞섬, < 0: 공정 지연
    CV  (Cost Variance)     = EV − AC   > 0: 원가 절감, < 0: 원가 초과

    SPI (Schedule Performance Index) = EV / PV   1.0 기준
    CPI (Cost Performance Index)     = EV / AC   1.0 기준

    ── 예측 지표 (잔여 공사 전망) ──
    BAC (Budget at Completion) = PV 전체 합계 (프로젝트 총예산)
    EAC (Estimate at Completion) = BAC / CPI    → 현재 CPI 추세로 완공 시 총비용
    ETC (Estimate to Complete)   = EAC − AC     → 앞으로 더 필요한 비용
    VAC (Variance at Completion) = BAC − EAC    → 완공 시점의 예상 예산 차이
    """

    result = df.copy()

    # ── 핵심 3대 지표 ──
    result["PV"] = result["planned_qty"]    * result["std_unit_price"]
    result["EV"] = result["actual_qty"]     * result["std_unit_price"]
    result["AC"] = result["actual_qty"]     * result["actual_unit_price"]

    # ── 차이(Variance) ──
    result["SV"] = result["EV"] - result["PV"]   # 공정 차이
    result["CV"] = result["EV"] - result["AC"]   # 원가 차이

    # ── 성과지수(Index) ──
    result["SPI"] = (result["EV"] / result["PV"]).replace([np.inf, -np.inf], np.nan).round(3)
    result["CPI"] = (result["EV"] / result["AC"]).replace([np.inf, -np.inf], np.nan).round(3)

    # ── 예측 지표 ──
    BAC = result["PV"].sum()
    result["EAC"] = (BAC / result["CPI"]).round(0)   # 행별 CPI로 전체 완공비 예측
    result["ETC"] = result["EAC"] - result["AC"]
    result["VAC"] = BAC - result["EAC"]

    # ── 공정 지연이 예산에 미치는 영향 분석 ──
    # SV < 0 (공정 지연) → 지연 물량이 추후 집중 투입 → 단가 상승 압력 발생
    # 영향 크기 = |SV| × (실제단가 / 기준단가 − 1)  : 지연 물량에 단가 프리미엄 적용
    price_premium = (result["actual_unit_price"] / result["std_unit_price"] - 1).clip(lower=0)
    result["delay_budget_impact"] = np.where(
        result["SV"] < 0,
        result["SV"].abs() * price_premium,   # 지연 시 추가 비용 압력 (양수 = 위험)
        0.0,
    ).round(0)

    # ── 상태 판정 ──
    result["SV_status"] = result["SV"].apply(
        lambda v: "공정 앞섬" if v > 0 else ("정상" if v == 0 else "공정 지연")
    )
    result["CV_status"] = result["CV"].apply(
        lambda v: "원가 절감" if v > 0 else ("정상" if v == 0 else "원가 초과")
    )
    result["SPI_grade"] = result["SPI"].apply(
        lambda v: "우수(≥1.0)" if (v is not None and v >= 1.0) else "위험(<1.0)"
    )
    result["CPI_grade"] = result["CPI"].apply(
        lambda v: "우수(≥1.0)" if (v is not None and v >= 1.0) else "위험(<1.0)"
    )

    return result


# ============================================================
# Step 3. 보고서 출력
# ============================================================

def print_evms_report(evms_df: pd.DataFrame):
    print("=" * 80)
    print("[ EVMS 분석 보고서 ]")
    print("=" * 80)

    display_cols = [
        "wbs_code", "work_name", "material_name", "unit",
        "planned_qty", "actual_qty",
        "PV", "EV", "AC",
        "SV", "CV",
        "SPI", "CPI",
        "SV_status", "CV_status",
        "delay_budget_impact",
    ]
    col_labels = [
        "WBS", "공종", "자재", "단위",
        "계획수량", "실적수량",
        "PV(원)", "EV(원)", "AC(원)",
        "SV(원)", "CV(원)",
        "SPI", "CPI",
        "공정상태", "원가상태",
        "지연→예산영향(원)",
    ]
    out = evms_df[display_cols].copy()
    out.columns = col_labels
    print(out.to_string(index=False))

    # 집계 요약
    BAC = evms_df["PV"].sum()
    total_EV  = evms_df["EV"].sum()
    total_AC  = evms_df["AC"].sum()
    total_SV  = evms_df["SV"].sum()
    total_CV  = evms_df["CV"].sum()
    total_SPI = total_EV / evms_df["PV"].sum()
    total_CPI = total_EV / total_AC
    EAC_proj  = BAC / total_CPI
    VAC_proj  = BAC - EAC_proj
    total_delay_impact = evms_df["delay_budget_impact"].sum()

    print("\n" + "─" * 80)
    print("[ 프로젝트 전체 요약 ]")
    print(f"  BAC (총 계획예산)       : {BAC:>15,.0f} 원")
    print(f"  총 PV                   : {BAC:>15,.0f} 원")
    print(f"  총 EV                   : {total_EV:>15,.0f} 원")
    print(f"  총 AC                   : {total_AC:>15,.0f} 원")
    print(f"  SV (공정 차이)          : {total_SV:>+15,.0f} 원  {'▲앞섬' if total_SV>0 else '▼지연'}")
    print(f"  CV (원가 차이)          : {total_CV:>+15,.0f} 원  {'▲절감' if total_CV>0 else '▼초과'}")
    print(f"  SPI (공정성과지수)      : {total_SPI:>15.3f}  {'정상' if total_SPI>=1 else '지연'}")
    print(f"  CPI (원가성과지수)      : {total_CPI:>15.3f}  {'정상' if total_CPI>=1 else '초과'}")
    print(f"  EAC (완공 예상비용)     : {EAC_proj:>15,.0f} 원")
    print(f"  VAC (완공 예산 차이)    : {VAC_proj:>+15,.0f} 원")
    print(f"  공정지연→예산 추가압력  : {total_delay_impact:>15,.0f} 원")
    print("─" * 80)


# ============================================================
# Step 4. 시각화
# ============================================================

def plot_evms(evms_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("EVMS 분석 대시보드", fontsize=16, fontweight="bold")

    labels = [f"{r['material_name']}\n({r['wbs_code']})" for _, r in evms_df.iterrows()]
    x = np.arange(len(labels))
    w = 0.25

    # ── 차트 1: PV / EV / AC 비교 ──────────────────────────
    ax = axes[0, 0]
    ax.bar(x - w, evms_df["PV"] / 1e6, w, label="PV (계획)", color="#4C72B0", alpha=0.85)
    ax.bar(x,     evms_df["EV"] / 1e6, w, label="EV (실적가치)", color="#55A868", alpha=0.85)
    ax.bar(x + w, evms_df["AC"] / 1e6, w, label="AC (실제비용)", color="#DD8452", alpha=0.85)
    ax.set_title("PV / EV / AC 비교", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("금액 (백만원)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}M"))
    ax.legend(fontsize=8)

    # ── 차트 2: SV / CV (차이 막대) ─────────────────────────
    ax = axes[0, 1]
    sv_colors = ["#55A868" if v >= 0 else "#C44E52" for v in evms_df["SV"]]
    cv_colors = ["#55A868" if v >= 0 else "#C44E52" for v in evms_df["CV"]]
    ax.bar(x - w/2, evms_df["SV"] / 1e3, w, color=sv_colors, alpha=0.85, label="SV (공정차이)")
    ax.bar(x + w/2, evms_df["CV"] / 1e3, w, color=cv_colors, alpha=0.6,  label="CV (원가차이)", hatch="//")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("SV (공정차이) / CV (원가차이)", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("차이 (천원)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}K"))
    ax.legend(fontsize=8)

    # ── 차트 3: SPI / CPI 성과지수 ──────────────────────────
    ax = axes[1, 0]
    ax.plot(x, evms_df["SPI"], marker="s", color="#4C72B0", linewidth=2,
            markersize=8, label="SPI (공정성과)")
    ax.plot(x, evms_df["CPI"], marker="o", color="#DD8452", linewidth=2,
            markersize=8, label="CPI (원가성과)")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, label="기준선 (1.0)")
    ax.axhline(0.9, color="#C44E52", linestyle=":", linewidth=1, alpha=0.7, label="경고선 (0.9)")
    ax.set_title("SPI / CPI 성과지수", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Index")
    ax.set_ylim(0.7, 1.3)
    ax.legend(fontsize=8)
    for i, (spi, cpi) in enumerate(zip(evms_df["SPI"], evms_df["CPI"])):
        ax.text(i, spi + 0.02, f"{spi:.2f}", ha="center", fontsize=7, color="#4C72B0")
        ax.text(i, cpi - 0.04, f"{cpi:.2f}", ha="center", fontsize=7, color="#DD8452")

    # ── 차트 4: 공정 지연 → 예산 영향 + EAC vs BAC ──────────
    ax = axes[1, 1]
    BAC = evms_df["PV"].sum()
    ax.bar(x, evms_df["delay_budget_impact"] / 1e3, color="#C44E52", alpha=0.7,
           label="지연→예산 추가압력")
    ax2 = ax.twinx()
    ax2.plot(x, evms_df["EAC"] / 1e6, marker="D", color="#9467BD", linewidth=2,
             markersize=7, label="EAC (행별완공예상)")
    ax2.axhline(BAC / 1e6, color="#4C72B0", linestyle="--", linewidth=1.5, label=f"BAC ({BAC/1e6:.1f}M)")
    ax.set_title("공정 지연의 예산 영향 & EAC 예측", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("추가 예산 압력 (천원)", color="#C44E52")
    ax2.set_ylabel("완공 예상비용 (백만원)", color="#9467BD")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

    plt.tight_layout()
    plt.savefig("EVMS_대시보드.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("→ 'EVMS_대시보드.png' 저장 완료")


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    base_df  = build_base_df(plan_df, vision_df, cost_df)
    evms_df  = compute_evms(base_df)

    print_evms_report(evms_df)

    # CSV 저장
    save_cols = [
        "capture_date", "wbs_code", "work_name", "material_name", "unit",
        "planned_qty", "actual_qty",
        "PV", "EV", "AC", "SV", "CV", "SPI", "CPI",
        "EAC", "ETC", "VAC",
        "SV_status", "CV_status", "SPI_grade", "CPI_grade",
        "delay_budget_impact",
    ]
    evms_df[save_cols].to_csv("EVMS_분석결과.csv", index=False, encoding="utf-8-sig")
    print("→ 'EVMS_분석결과.csv' 저장 완료\n")

    plot_evms(evms_df)
