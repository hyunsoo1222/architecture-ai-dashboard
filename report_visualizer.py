"""
자재관리 통합 보고서 시각화
material_management_db.py 실행 후 생성된 데이터를 차트로 출력
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np

# 한글 폰트 설정 (Windows)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 보고서 데이터 로드 (또는 직접 생성)
# ============================================================

def load_report() -> pd.DataFrame:
    try:
        df = pd.read_csv("자재관리_통합보고서.csv", encoding="utf-8-sig")
        df["실적일자"] = pd.to_datetime(df["실적일자"])
        return df
    except FileNotFoundError:
        # material_management_db.py 의 데이터 인라인 재현
        from material_management_db import report_df
        return report_df


report_df = load_report()

# ============================================================
# 색상 팔레트
# ============================================================
COLOR_PLAN   = "#4C72B0"
COLOR_ACTUAL = "#DD8452"
COLOR_OK     = "#55A868"
COLOR_WARN   = "#FFC107"
COLOR_ALERT  = "#C44E52"

STATUS_COLOR = {"정상": COLOR_OK, "주의": COLOR_WARN, "경고": COLOR_ALERT, "매칭없음": "#AAAAAA"}

# ============================================================
# Figure 생성 (2×3 레이아웃)
# ============================================================
fig = plt.figure(figsize=(18, 12))
fig.suptitle("건설 현장 자재관리 통합 보고서", fontsize=18, fontweight="bold", y=0.98)

# ── 차트 1: 자재별 계획 vs 실적 수량 (묶음 막대) ───────────────
ax1 = fig.add_subplot(2, 3, 1)
materials = report_df["자재명"].unique()
x = np.arange(len(materials))
width = 0.35

plan_qty   = [report_df[report_df["자재명"] == m]["계획수량"].sum() for m in materials]
actual_qty = [report_df[report_df["자재명"] == m]["실적수량"].sum() for m in materials]

bars1 = ax1.bar(x - width/2, plan_qty,   width, label="계획수량", color=COLOR_PLAN,   alpha=0.85)
bars2 = ax1.bar(x + width/2, actual_qty, width, label="실적수량", color=COLOR_ACTUAL, alpha=0.85)

ax1.set_title("자재별 계획 vs 실적 수량", fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(materials, fontsize=9)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax1.legend(fontsize=8)
ax1.set_ylabel("수량 (단위 혼합)")

for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
             f"{bar.get_height():,.0f}", ha="center", va="bottom", fontsize=7)
for bar in bars2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
             f"{bar.get_height():,.0f}", ha="center", va="bottom", fontsize=7)

# ── 차트 2: 자재별 달성률 (가로 막대 + 목표선) ────────────────
ax2 = fig.add_subplot(2, 3, 2)
rates = [report_df[report_df["자재명"] == m]["달성률(%)"].mean() for m in materials]
colors = [COLOR_OK if r >= 95 else COLOR_WARN if r >= 80 else COLOR_ALERT for r in rates]

bars = ax2.barh(materials, rates, color=colors, alpha=0.85, edgecolor="white")
ax2.axvline(100, color="gray", linestyle="--", linewidth=1.2, label="목표(100%)")
ax2.axvline(80,  color=COLOR_ALERT, linestyle=":",  linewidth=1, alpha=0.7, label="경고선(80%)")
ax2.set_xlim(0, 130)
ax2.set_title("자재별 달성률 (%)", fontweight="bold")
ax2.set_xlabel("달성률 (%)")
ax2.legend(fontsize=7)

for bar, rate in zip(bars, rates):
    ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
             f"{rate:.1f}%", va="center", fontsize=9, fontweight="bold")

# ── 차트 3: 계획 vs 실적 금액 (묶음 막대) ────────────────────
ax3 = fig.add_subplot(2, 3, 3)
plan_cost   = [report_df[report_df["자재명"] == m]["계획금액"].sum()/1e6 for m in materials]
actual_cost = [report_df[report_df["자재명"] == m]["실적금액"].sum()/1e6 for m in materials]

ax3.bar(x - width/2, plan_cost,   width, label="계획금액", color=COLOR_PLAN,   alpha=0.85)
ax3.bar(x + width/2, actual_cost, width, label="실적금액", color=COLOR_ACTUAL, alpha=0.85)
ax3.set_title("자재별 계획 vs 실적 금액", fontweight="bold")
ax3.set_xticks(x)
ax3.set_xticklabels(materials, fontsize=9)
ax3.set_ylabel("금액 (백만원)")
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}M"))
ax3.legend(fontsize=8)

# ── 차트 4: 단가 증감률 (색상으로 과/부족 표시) ───────────────
ax4 = fig.add_subplot(2, 3, 4)
price_vars = report_df["단가증감률(%)"].fillna(0)
bar_colors = [COLOR_ALERT if v > 0 else COLOR_OK for v in price_vars]

ax4.bar(range(len(report_df)), price_vars, color=bar_colors, alpha=0.85)
ax4.axhline(0, color="black", linewidth=0.8)
ax4.set_title("행별 단가 증감률 (%)", fontweight="bold")
ax4.set_ylabel("증감률 (%)")
ax4.set_xticks(range(len(report_df)))
ax4.set_xticklabels(
    [f"{r['자재명']}\n{str(r['실적일자'])[:7]}" for _, r in report_df.iterrows()],
    fontsize=7, rotation=30, ha="right"
)

over_patch  = mpatches.Patch(color=COLOR_ALERT, alpha=0.85, label="단가 상승 (원가 초과)")
under_patch = mpatches.Patch(color=COLOR_OK,    alpha=0.85, label="단가 절감")
ax4.legend(handles=[over_patch, under_patch], fontsize=7)

# ── 차트 5: AI 신뢰도 분포 (꺾은선) ─────────────────────────
ax5 = fig.add_subplot(2, 3, 5)
ax5.plot(range(len(report_df)), report_df["AI신뢰도"],
         marker="o", color="#9467BD", linewidth=2, markersize=7)
ax5.axhline(0.90, color=COLOR_ALERT, linestyle="--", linewidth=1, label="신뢰도 기준(0.90)")
ax5.set_ylim(0.75, 1.0)
ax5.set_title("항목별 AI 신뢰도", fontweight="bold")
ax5.set_ylabel("Confidence")
ax5.set_xticks(range(len(report_df)))
ax5.set_xticklabels(
    [f"{r['자재명']}" for _, r in report_df.iterrows()],
    fontsize=8, rotation=30, ha="right"
)
ax5.legend(fontsize=8)

for i, v in enumerate(report_df["AI신뢰도"]):
    ax5.text(i, v + 0.003, f"{v:.2f}", ha="center", fontsize=7)

# ── 차트 6: 상태 파이차트 + 요약 텍스트 ──────────────────────
ax6 = fig.add_subplot(2, 3, 6)
status_counts = report_df["상태"].value_counts()
pie_colors = [STATUS_COLOR.get(s, "#AAAAAA") for s in status_counts.index]

wedges, texts, autotexts = ax6.pie(
    status_counts.values,
    labels=status_counts.index,
    colors=pie_colors,
    autopct="%1.0f%%",
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
)
for at in autotexts:
    at.set_fontsize(10)
ax6.set_title("관리 상태 분포", fontweight="bold")

# 요약 텍스트 박스
total_plan   = report_df["계획금액"].sum()
total_actual = report_df["실적금액"].sum()
diff         = total_actual - total_plan
summary_text = (
    f"총 계획금액: {total_plan/1e6:,.1f}M원\n"
    f"총 실적금액: {total_actual/1e6:,.1f}M원\n"
    f"금액 차이:  {diff/1e6:+,.1f}M원\n"
    f"전체 달성률: {total_actual/total_plan*100:.1f}%"
)
ax6.text(0, -1.55, summary_text, transform=ax6.transAxes,
         fontsize=9, va="top", ha="left",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0F0F0", edgecolor="#CCCCCC"))

# ============================================================
# 저장 및 출력
# ============================================================
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("자재관리_보고서_차트.png", dpi=150, bbox_inches="tight")
plt.show()
print("→ '자재관리_보고서_차트.png' 저장 완료")
