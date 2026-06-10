"""
건설 현장 자재관리 자동화 시스템 - DB 설계 및 병합
건축 AI 수업 프로젝트 - 모듈2 7조
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================
# 1. PLAN DB (계획) - 공정/WBS 기반 자재 계획
# ============================================================

plan_data = {
    "wbs_code":       ["A-01", "A-01", "A-02", "A-02", "B-01", "B-01", "B-02"],
    "work_name":      ["기초공사", "기초공사", "골조공사", "골조공사", "지하1층 골조", "지하1층 골조", "지상1층 골조"],
    "material_name":  ["철근", "레미콘", "철근", "레미콘", "철골", "볼트", "철근"],
    "planned_date":   ["2024-03-01", "2024-03-01", "2024-04-01", "2024-04-01",
                       "2024-04-15", "2024-04-15", "2024-05-01"],
    "planned_qty":    [5000, 120, 8000, 200, 3000, 500, 6000],   # 단위: kg, m³, kg, m³, kg, EA, kg
    "unit":           ["kg", "m³", "kg", "m³", "kg", "EA", "kg"],
    "std_unit_price": [1050, 85000, 1050, 85000, 2200, 800, 1050],  # 원/단위
    "location":       ["B1-기초", "B1-기초", "B1-골조", "B1-골조", "B1-전체", "B1-전체", "1F-골조"],
}

plan_df = pd.DataFrame(plan_data)
plan_df["planned_date"] = pd.to_datetime(plan_df["planned_date"])
plan_df["planned_cost"] = plan_df["planned_qty"] * plan_df["std_unit_price"]

print("=" * 60)
print("[ PLAN DB - 자재 계획 ]")
print("=" * 60)
print(plan_df.to_string(index=False))
print(f"\n총 계획 예산: {plan_df['planned_cost'].sum():,.0f} 원\n")


# ============================================================
# 2. VISION AI DB (실적) - AI 인식 결과 (SODA 데이터셋 기반 가정)
# ============================================================
# Bounding Box 카운트 → 수량 환산 로직 포함
# 철근: 1 bbox ≈ 50kg, 레미콘: 차량 1대 ≈ 6m³, 철골: 1 bbox ≈ 200kg

BBOX_TO_QTY = {
    "철근":  50,    # kg/bbox
    "레미콘": 6,    # m³/vehicle
    "철골":  200,   # kg/bbox
    "볼트":  10,    # EA/bbox
}

vision_data = {
    "capture_date":    ["2024-03-05", "2024-03-05", "2024-04-03", "2024-04-03",
                        "2024-04-18", "2024-04-18", "2024-05-02"],
    "location":        ["B1-기초", "B1-기초", "B1-골조", "B1-골조",
                        "B1-전체", "B1-전체", "1F-골조"],
    "material_name":   ["철근", "레미콘", "철근", "레미콘", "철골", "볼트", "철근"],
    "bbox_count":      [92, 18, 148, 31, 14, 46, 110],
    "confidence_avg":  [0.91, 0.88, 0.93, 0.85, 0.90, 0.87, 0.92],  # AI 신뢰도
    "image_count":     [12, 8, 20, 10, 15, 15, 18],                   # 사용된 이미지 수
}

vision_df = pd.DataFrame(vision_data)
vision_df["capture_date"] = pd.to_datetime(vision_df["capture_date"])

# bbox → 실제 수량 환산
vision_df["unit_per_bbox"] = vision_df["material_name"].map(BBOX_TO_QTY)
vision_df["actual_qty"] = vision_df["bbox_count"] * vision_df["unit_per_bbox"]

print("=" * 60)
print("[ VISION AI DB - AI 인식 실적 ]")
print("=" * 60)
print(vision_df[["capture_date", "location", "material_name",
                  "bbox_count", "confidence_avg", "actual_qty"]].to_string(index=False))
print()


# ============================================================
# 3. COST DB (비용) - 실제 구매 단가 (시장가 연동 가정)
# ============================================================

cost_data = {
    "price_date":      ["2024-03-01", "2024-04-01", "2024-04-01", "2024-04-15",
                        "2024-05-01", "2024-03-01", "2024-04-15"],
    "material_name":   ["철근", "철근", "레미콘", "철골", "철근", "레미콘", "볼트"],
    "actual_unit_price": [1080, 1100, 87000, 2350, 1090, 86000, 820],  # 원/단위
    "supplier":        ["대한철강", "대한철강", "삼표레미콘", "현대제철",
                        "동국제강", "아세아시멘트", "대명볼트"],
    "market_index":    [102.9, 104.8, 102.4, 106.8, 103.8, 101.2, 102.5],  # 물가지수
}

cost_df = pd.DataFrame(cost_data)
cost_df["price_date"] = pd.to_datetime(cost_df["price_date"])

print("=" * 60)
print("[ COST DB - 실제 구매 단가 ]")
print("=" * 60)
print(cost_df.to_string(index=False))
print()


# ============================================================
# 4. DB 병합 (날짜 + 자재명 기준 Merge)
# ============================================================

def merge_dbs(plan_df, vision_df, cost_df, date_tolerance_days=7):
    """
    세 DB를 날짜(±허용오차)와 자재명을 기준으로 병합.
    날짜 허용오차: 계획일 ±N일 이내의 실적/단가를 매칭
    """

    # Step 1: Vision AI + Cost DB 병합 (자재명 기준 asof merge)
    vision_sorted = vision_df.sort_values("capture_date")
    cost_sorted = cost_df.sort_values("price_date")

    # 자재별로 가장 가까운 이전 단가 매칭 (pd.merge_asof)
    actual_cost_df = pd.merge_asof(
        vision_sorted,
        cost_sorted.rename(columns={"price_date": "capture_date"}),
        on="capture_date",
        by="material_name",
        direction="nearest",         # 가장 가까운 날짜의 단가 사용
        tolerance=pd.Timedelta(f"{date_tolerance_days}D"),
    )
    actual_cost_df["actual_cost"] = (
        actual_cost_df["actual_qty"] * actual_cost_df["actual_unit_price"]
    )

    # Step 2: Plan DB + (Vision+Cost) 병합
    # 날짜 허용오차를 위한 날짜 범위 키 생성
    plan_sorted = plan_df.sort_values("planned_date")

    merged = pd.merge_asof(
        actual_cost_df.sort_values("capture_date"),
        plan_sorted[["planned_date", "wbs_code", "work_name", "material_name",
                     "planned_qty", "std_unit_price", "planned_cost", "unit"]],
        left_on="capture_date",
        right_on="planned_date",
        by="material_name",
        direction="nearest",
        tolerance=pd.Timedelta(f"{date_tolerance_days}D"),
    )

    return merged


merged_df = merge_dbs(plan_df, vision_df, cost_df)

# ============================================================
# 5. 분석 컬럼 생성 (달성률, 차이 분석)
# ============================================================

merged_df["qty_variance"] = merged_df["actual_qty"] - merged_df["planned_qty"]
merged_df["qty_achievement_rate"] = (
    merged_df["actual_qty"] / merged_df["planned_qty"] * 100
).round(1)

merged_df["cost_variance"] = merged_df["actual_cost"] - merged_df["planned_cost"]
merged_df["price_variance_rate"] = (
    (merged_df["actual_unit_price"] - merged_df["std_unit_price"])
    / merged_df["std_unit_price"] * 100
).round(1)

# 상태 판정
def classify_status(row):
    if pd.isna(row["planned_qty"]):
        return "매칭없음"
    rate = row["qty_achievement_rate"]
    if rate >= 95:
        return "정상"
    elif rate >= 80:
        return "주의"
    else:
        return "경고"

merged_df["status"] = merged_df.apply(classify_status, axis=1)

# ============================================================
# 6. 최종 통합 보고서 출력
# ============================================================

report_cols = [
    "capture_date", "wbs_code", "work_name", "material_name", "unit",
    "planned_qty", "actual_qty", "qty_achievement_rate",
    "std_unit_price", "actual_unit_price", "price_variance_rate",
    "planned_cost", "actual_cost", "cost_variance",
    "confidence_avg", "status",
]

report_df = merged_df[report_cols].copy()
report_df.columns = [
    "실적일자", "WBS코드", "공종명", "자재명", "단위",
    "계획수량", "실적수량", "달성률(%)",
    "기준단가", "실제단가", "단가증감률(%)",
    "계획금액", "실적금액", "금액차이",
    "AI신뢰도", "상태",
]

print("=" * 60)
print("[ 통합 자재관리 보고서 ]")
print("=" * 60)
print(report_df.to_string(index=False))
print()

# 요약 집계
print("=" * 60)
print("[ 요약 집계 ]")
print("=" * 60)
summary = report_df.groupby("자재명").agg(
    계획수량합=("계획수량", "sum"),
    실적수량합=("실적수량", "sum"),
    계획금액합=("계획금액", "sum"),
    실적금액합=("실적금액", "sum"),
    평균AI신뢰도=("AI신뢰도", "mean"),
).round(2)
summary["금액차이"] = summary["실적금액합"] - summary["계획금액합"]
summary["달성률(%)"] = (summary["실적수량합"] / summary["계획수량합"] * 100).round(1)
print(summary.to_string())

total_plan = report_df["계획금액"].sum()
total_actual = report_df["실적금액"].sum()
print(f"\n총 계획금액: {total_plan:>15,.0f} 원")
print(f"총 실적금액: {total_actual:>15,.0f} 원")
print(f"총 금액차이: {total_actual - total_plan:>+15,.0f} 원")
print(f"전체 달성률: {total_actual / total_plan * 100:>14.1f} %")

# CSV 저장
report_df.to_csv("자재관리_통합보고서.csv", index=False, encoding="utf-8-sig")
print("\n→ '자재관리_통합보고서.csv' 저장 완료")
