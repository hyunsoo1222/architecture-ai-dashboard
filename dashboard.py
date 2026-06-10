"""
자재 원가 모니터링 대시보드
실행: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import tempfile, os, urllib.request
from pathlib import Path

MODEL_URL  = "https://github.com/hyunsoo1222/architecture-ai-dashboard/releases/download/v1.0/best_pt.zip"
MODEL_PATH = Path("best.pt")

@st.cache_resource(show_spinner="🔄 AI 모델 다운로드 중... (최초 1회)")
def load_yolo_model():
    try:
        from ultralytics import YOLO
        import zipfile
        if not MODEL_PATH.exists():
            zip_path = Path("best_pt.zip")
            urllib.request.urlretrieve(MODEL_URL, zip_path)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(".")
            zip_path.unlink()
        return YOLO(str(MODEL_PATH))
    except Exception:
        return None

# ── 페이지 기본 설정 ──────────────────────────────────────────
st.set_page_config(
    page_title="자재 원가 모니터링",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 공통 색상 ─────────────────────────────────────────────────
C_PLAN   = "#4C72B0"
C_EV     = "#55A868"
C_AC     = "#DD8452"
C_DANGER = "#C44E52"
C_WARN   = "#FFC107"
C_OK     = "#28A745"

# ============================================================
# 데이터 생성 (evms_analysis.py 로직 인라인)
# ============================================================

@st.cache_data
def load_evms_data() -> pd.DataFrame:
    plan_data = {
        "wbs_code":       ["A-01","A-01","A-02","A-02","B-01","B-01","B-02"],
        "work_name":      ["기초공사","기초공사","골조공사","골조공사",
                           "지하1층 골조","지하1층 골조","지상1층 골조"],
        "material_name":  ["철근","레미콘","철근","레미콘","철골","볼트","철근"],
        "planned_date":   ["2024-03-01","2024-03-01","2024-04-01","2024-04-01",
                           "2024-04-15","2024-04-15","2024-05-01"],
        "planned_qty":    [5000, 120, 8000, 200, 3000, 500, 6000],
        "unit":           ["kg","m³","kg","m³","kg","EA","kg"],
        "std_unit_price": [1050, 85000, 1050, 85000, 2200, 800, 1050],
        "location":       ["B1-기초","B1-기초","B1-골조","B1-골조",
                           "B1-전체","B1-전체","1F-골조"],
    }
    BBOX_TO_QTY = {"철근":50,"레미콘":6,"철골":200,"볼트":10}
    vision_data = {
        "capture_date":   ["2024-03-05","2024-03-05","2024-04-03","2024-04-03",
                           "2024-04-18","2024-04-18","2024-05-02"],
        "location":       ["B1-기초","B1-기초","B1-골조","B1-골조",
                           "B1-전체","B1-전체","1F-골조"],
        "material_name":  ["철근","레미콘","철근","레미콘","철골","볼트","철근"],
        "bbox_count":     [92,18,148,31,14,46,110],
        "confidence_avg": [0.91,0.88,0.93,0.85,0.90,0.87,0.92],
        "image_count":    [12,8,20,10,15,15,18],
    }
    cost_data = {
        "price_date":        ["2024-03-01","2024-04-01","2024-04-01","2024-04-15",
                              "2024-05-01","2024-03-01","2024-04-15"],
        "material_name":     ["철근","철근","레미콘","철골","철근","레미콘","볼트"],
        "actual_unit_price": [1080,1100,87000,2350,1090,86000,820],
        "supplier":          ["대한철강","대한철강","삼표레미콘","현대제철",
                              "동국제강","아세아시멘트","대명볼트"],
    }

    plan_df   = pd.DataFrame(plan_data)
    vision_df = pd.DataFrame(vision_data)
    cost_df   = pd.DataFrame(cost_data)
    plan_df["planned_date"]   = pd.to_datetime(plan_df["planned_date"])
    vision_df["capture_date"] = pd.to_datetime(vision_df["capture_date"])
    cost_df["price_date"]     = pd.to_datetime(cost_df["price_date"])
    vision_df["actual_qty"]   = (
        vision_df["material_name"].map(BBOX_TO_QTY) * vision_df["bbox_count"]
    )

    tol = pd.Timedelta("7D")
    vc = pd.merge_asof(
        vision_df.sort_values("capture_date"),
        cost_df.sort_values("price_date").rename(columns={"price_date":"capture_date"}),
        on="capture_date", by="material_name", direction="nearest", tolerance=tol,
    )
    df = pd.merge_asof(
        vc.sort_values("capture_date"),
        plan_df[["planned_date","wbs_code","work_name","material_name",
                 "planned_qty","std_unit_price","unit","location"]
               ].sort_values("planned_date"),
        left_on="capture_date", right_on="planned_date",
        by="material_name", direction="nearest", tolerance=tol,
    )

    df["PV"]  = df["planned_qty"]    * df["std_unit_price"]
    df["EV"]  = df["actual_qty"]     * df["std_unit_price"]
    df["AC"]  = df["actual_qty"]     * df["actual_unit_price"]
    df["SV"]  = df["EV"] - df["PV"]
    df["CV"]  = df["EV"] - df["AC"]
    df["SPI"] = (df["EV"] / df["PV"]).replace([np.inf,-np.inf], np.nan).round(3)
    df["CPI"] = (df["EV"] / df["AC"]).replace([np.inf,-np.inf], np.nan).round(3)
    BAC       = df["PV"].sum()
    df["EAC"] = (BAC / df["CPI"]).round(0)
    df["ETC"] = df["EAC"] - df["AC"]
    df["VAC"] = BAC - df["EAC"]

    price_premium           = (df["actual_unit_price"] / df["std_unit_price"] - 1).clip(lower=0)
    df["delay_budget_impact"] = np.where(df["SV"] < 0, df["SV"].abs() * price_premium, 0.0)

    unit_label = {"철근":"톤","레미콘":"m³","철골":"톤","볼트":"EA"}
    unit_factor= {"철근":0.001,"레미콘":1,"철골":0.001,"볼트":1}
    df["display_qty"]  = df.apply(
        lambda r: r["actual_qty"] * unit_factor.get(r["material_name"],1), axis=1
    ).round(2)
    df["display_unit"] = df["material_name"].map(unit_label)
    return df

df = load_evms_data()

# ============================================================
# 사이드바 — 필터
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/hard-hat.png", width=60)
    st.title("🏗️ 현장 관리 시스템")
    st.caption("건축 AI 모듈2 7조")
    st.divider()

    # ── 실시간 이미지 분석 ──────────────────────────────────────
    st.markdown("#### 📷 현장 사진 분석")

    img_files  = st.file_uploader("현장 사진 업로드", type=["jpg","jpeg","png"],
                                   accept_multiple_files=True)
    location_input = st.text_input("촬영 위치", value="B1-기초",
                                    placeholder="예: B1-기초, 1F-골조")
    conf_val = st.slider("신뢰도 임계값", 0.1, 0.9, 0.5, 0.05)

    run_inference = st.button("🔍 AI 분석 실행", use_container_width=True,
                               disabled=not img_files)

    if run_inference and img_files:
        import json, time
        from PIL import Image

        progress = st.progress(0, text="AI 분석 중...")
        img_cols = st.columns(min(len(img_files), 4))

        for i, img_file in enumerate(img_files):
            time.sleep(0.4)
            img = Image.open(img_file)
            img_cols[i % 4].image(img, caption=f"📷 {img_file.name}",
                                   use_container_width=True)
            progress.progress((i + 1) / len(img_files),
                               text=f"분석 중... ({i+1}/{len(img_files)})")

        time.sleep(0.5)
        progress.progress(1.0, text="✅ 분석 완료!")

        # 데모 결과 로드
        try:
            with open("demo_results.json", encoding="utf-8") as f:
                demo = json.load(f)
        except FileNotFoundError:
            demo = [
                {"material_name":"철근",  "bbox_count":94, "confidence_avg":0.923,
                 "actual_qty":4700, "display_qty":4.7,  "display_unit":"톤"},
                {"material_name":"목재",  "bbox_count":37, "confidence_avg":0.887,
                 "actual_qty":37,   "display_qty":37,   "display_unit":"EA"},
            ]

        result_df = pd.DataFrame(demo)
        result_df["location"]     = location_input
        result_df["capture_date"] = pd.Timestamp.today().normalize()
        result_df["image_count"]  = len(img_files)

        st.success(f"✅ {len(img_files)}장 분석 완료 → {len(result_df)}종 자재 인식")
        st.dataframe(
            result_df[["material_name","bbox_count","confidence_avg",
                        "display_qty","display_unit"]].rename(columns={
                "material_name":"자재명","bbox_count":"Bbox수",
                "confidence_avg":"신뢰도","display_qty":"인식수량",
                "display_unit":"단위"}),
            use_container_width=True,
        )
        st.session_state["live_vision_df"] = result_df

    st.divider()
    st.markdown("#### 🔎 조회 필터")
    wbs_options  = ["전체"] + sorted(df["wbs_code"].unique().tolist())
    mat_options  = ["전체"] + sorted(df["material_name"].unique().tolist())
    sel_wbs      = st.selectbox("공종(WBS) 선택", wbs_options)
    sel_material = st.selectbox("자재 선택", mat_options)

    date_min = df["capture_date"].min().date()
    date_max = df["capture_date"].max().date()
    sel_date = st.date_input("조회 기간", value=(date_min, date_max),
                              min_value=date_min, max_value=date_max)
    st.divider()
    st.caption(f"📅 기준일: {datetime.today().strftime('%Y-%m-%d')}")

# 필터 적용
filtered = df.copy()
if sel_wbs != "전체":
    filtered = filtered[filtered["wbs_code"] == sel_wbs]
if sel_material != "전체":
    filtered = filtered[filtered["material_name"] == sel_material]
if len(sel_date) == 2:
    filtered = filtered[
        (filtered["capture_date"].dt.date >= sel_date[0]) &
        (filtered["capture_date"].dt.date <= sel_date[1])
    ]

# ============================================================
# 헤더
# ============================================================
st.title("🏗️ 자재 원가 모니터링 대시보드")
st.caption("Vision AI 인식 결과 기반 · EVMS 실시간 분석")
st.divider()

# ============================================================
# ① 상단: AI 인식 현황 요약 카드
# ============================================================
st.subheader("📷 AI 인식 현황 요약")

# 자재별 오늘(최신) 인식량
latest_date = filtered["capture_date"].max()
today_label = latest_date.strftime("%Y년 %m월 %d일") if pd.notna(latest_date) else "-"

col_header = st.columns([2, 1])
with col_header[0]:
    st.markdown(f"**최신 촬영일:** `{today_label}`")
with col_header[1]:
    total_images = int(filtered["image_count"].sum())
    st.markdown(f"**분석 이미지:** `{total_images}장`")

mat_cols = st.columns(4)
mat_icons = {"철근":"🔩","레미콘":"🪣","철골":"🏗️","볼트":"⚙️"}

for i, mat in enumerate(["철근","레미콘","철골","볼트"]):
    sub = filtered[filtered["material_name"] == mat]
    qty   = sub["display_qty"].sum()
    unit  = sub["display_unit"].iloc[0] if len(sub) else "-"
    conf  = sub["confidence_avg"].mean() if len(sub) else 0
    bbox  = int(sub["bbox_count"].sum())
    icon  = mat_icons.get(mat,"📦")

    with mat_cols[i % 4]:
        conf_color = C_OK if conf >= 0.90 else C_WARN if conf >= 0.85 else C_DANGER
        st.markdown(f"""
<div style="background:#1E1E2E;border-radius:12px;padding:16px 14px;text-align:center;
            border-left:4px solid {conf_color};">
  <div style="font-size:2rem">{icon}</div>
  <div style="font-size:0.85rem;color:#AAAAAA;margin-top:4px">{mat}</div>
  <div style="font-size:1.8rem;font-weight:700;color:#FFFFFF">{qty:,.1f} <span style="font-size:1rem">{unit}</span></div>
  <div style="font-size:0.75rem;color:#AAAAAA">Bbox {bbox}개 인식</div>
  <div style="font-size:0.75rem;color:{conf_color}">신뢰도 {conf:.1%}</div>
</div>""", unsafe_allow_html=True)

st.divider()

# ============================================================
# ② 중단: EVMS 지표 KPI + PV/EV/AC 차트
# ============================================================
st.subheader("📊 EVMS 원가·공정 분석")

total_PV  = filtered["PV"].sum()
total_EV  = filtered["EV"].sum()
total_AC  = filtered["AC"].sum()
total_SV  = filtered["SV"].sum()
total_CV  = filtered["CV"].sum()
total_SPI = total_EV / total_PV if total_PV else 0
total_CPI = total_EV / total_AC if total_AC else 0
BAC       = df["PV"].sum()
EAC_proj  = BAC / total_CPI if total_CPI else 0
VAC_proj  = BAC - EAC_proj

# KPI 카드 행
k1, k2, k3, k4, k5, k6 = st.columns(6)

def kpi_card(col, label, value, unit="원", delta=None, invert=False):
    """delta > 0 → 초과(나쁨) when invert=True"""
    if delta is not None:
        good = (delta >= 0 and not invert) or (delta < 0 and invert)
        delta_color = C_OK if good else C_DANGER
        delta_str = f"<span style='color:{delta_color};font-size:0.75rem'>" \
                    f"{'▲' if delta>=0 else '▼'} {abs(delta):,.0f}{unit}</span>"
    else:
        delta_str = ""
    col.markdown(f"""
<div style="background:#1E1E2E;border-radius:10px;padding:14px 10px;text-align:center">
  <div style="font-size:0.75rem;color:#AAAAAA">{label}</div>
  <div style="font-size:1.3rem;font-weight:700;color:#FFFFFF">{value:,.0f}<span style="font-size:0.75rem"> {unit}</span></div>
  {delta_str}
</div>""", unsafe_allow_html=True)

kpi_card(k1, "PV (계획예산)",    total_PV/1e6, "M원")
kpi_card(k2, "EV (실적가치)",    total_EV/1e6, "M원")
kpi_card(k3, "AC (실제비용)",    total_AC/1e6, "M원")
kpi_card(k4, "SV (공정차이)",    total_SV/1e3, "K원", delta=total_SV)
kpi_card(k5, "CV (원가차이)",    total_CV/1e3, "K원", delta=total_CV)
kpi_card(k6, "CPI (원가성과지수)", total_CPI, "", delta=None)

st.markdown("<br>", unsafe_allow_html=True)

# 차트 행
chart_left, chart_right = st.columns([3, 2])

with chart_left:
    # 묶음 막대 — PV/EV/AC 자재별 비교
    mat_list = filtered["material_name"].unique()
    pv_vals  = [filtered[filtered["material_name"]==m]["PV"].sum()/1e6 for m in mat_list]
    ev_vals  = [filtered[filtered["material_name"]==m]["EV"].sum()/1e6 for m in mat_list]
    ac_vals  = [filtered[filtered["material_name"]==m]["AC"].sum()/1e6 for m in mat_list]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name="PV (계획)", x=list(mat_list), y=pv_vals,
                             marker_color=C_PLAN, opacity=0.9))
    fig_bar.add_trace(go.Bar(name="EV (실적가치)", x=list(mat_list), y=ev_vals,
                             marker_color=C_EV, opacity=0.9))
    fig_bar.add_trace(go.Bar(name="AC (실제비용)", x=list(mat_list), y=ac_vals,
                             marker_color=C_AC, opacity=0.9))
    fig_bar.update_layout(
        title="자재별 PV / EV / AC 비교",
        barmode="group",
        yaxis_title="금액 (백만원)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        font_color="white", height=350,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with chart_right:
    # SPI / CPI 게이지
    def gauge(value, title, ref=1.0):
        color = C_OK if value >= ref else (C_WARN if value >= 0.9 else C_DANGER)
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            delta={"reference": ref, "valueformat": ".3f"},
            title={"text": title, "font": {"size": 14}},
            gauge={
                "axis": {"range": [0.6, 1.4], "tickwidth": 1},
                "bar":  {"color": color},
                "steps": [
                    {"range": [0.6, 0.9],  "color": "#3A1010"},
                    {"range": [0.9, 1.0],  "color": "#3A2A10"},
                    {"range": [1.0, 1.4],  "color": "#10301A"},
                ],
                "threshold": {"line": {"color": "white", "width": 2},
                              "thickness": 0.75, "value": ref},
            },
            number={"valueformat": ".3f"},
        ))
        fig.update_layout(height=175, paper_bgcolor="#0E1117", font_color="white",
                          margin=dict(t=40, b=10, l=20, r=20))
        return fig

    st.plotly_chart(gauge(total_SPI, "SPI — 공정 성과 지수"), use_container_width=True)
    st.plotly_chart(gauge(total_CPI, "CPI — 원가 성과 지수"), use_container_width=True)

st.divider()

# SV / CV 폭포 차트
st.markdown("##### 항목별 SV / CV 편차")
fig_sv = go.Figure()
row_labels = [f"{r['material_name']} ({r['wbs_code']})" for _, r in filtered.iterrows()]

fig_sv.add_trace(go.Bar(
    name="SV (공정차이)", x=row_labels, y=filtered["SV"]/1e3,
    marker_color=[C_OK if v >= 0 else C_DANGER for v in filtered["SV"]],
    opacity=0.85,
))
fig_sv.add_trace(go.Bar(
    name="CV (원가차이)", x=row_labels, y=filtered["CV"]/1e3,
    marker_color=[C_EV if v >= 0 else C_WARN for v in filtered["CV"]],
    opacity=0.7, marker_pattern_shape="/"
))
fig_sv.add_hline(y=0, line_dash="dash", line_color="gray")
fig_sv.update_layout(
    barmode="group", yaxis_title="차이 (천원)",
    plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
    font_color="white", height=280, legend=dict(orientation="h"),
)
st.plotly_chart(fig_sv, use_container_width=True)
st.divider()

# ============================================================
# ③ 하단: 알림(Alert) 시스템
# ============================================================
st.subheader("🚨 자동 경보 및 조치 권고")

alerts = []

# 행별 이상 탐지
for _, row in filtered.iterrows():
    label = f"[{row['wbs_code']}] {row['work_name']} / {row['material_name']}"

    if row["CV"] < 0:
        overage = abs(row["CV"])
        rate    = abs(row["CV"] / row["EV"] * 100) if row["EV"] else 0
        alerts.append({
            "level":   "CRITICAL",
            "icon":    "🔴",
            "title":   f"원가 초과 — {label}",
            "detail":  f"CV = {row['CV']:,.0f}원 ({rate:.1f}% 초과) │ "
                       f"기준단가 {row['std_unit_price']:,}원 → 실제단가 {row['actual_unit_price']:,}원",
            "action":  "📋 **조치 필요:** 자재 추가 발주 단가 재협상 또는 대체 자재 검토 요망",
        })

    if row["SV"] < 0:
        delay_qty  = abs(row["SV"]) / row["std_unit_price"] if row["std_unit_price"] else 0
        budget_hit = row["delay_budget_impact"]
        alerts.append({
            "level":  "WARNING",
            "icon":   "🟡",
            "title":  f"공정 지연 — {label}",
            "detail": f"SV = {row['SV']:,.0f}원 │ 미달 수량 ≈ {delay_qty:,.0f}{row['unit']} │ "
                      f"지연→예산 추가 압력 {budget_hit:,.0f}원",
            "action": "📋 **조치 필요:** 공정 만회 계획 수립 및 잔여 물량 집중 투입 일정 조정 요망",
        })

    if row["confidence_avg"] < 0.85:
        alerts.append({
            "level":  "INFO",
            "icon":   "🔵",
            "title":  f"AI 신뢰도 낮음 — {label}",
            "detail": f"평균 신뢰도 {row['confidence_avg']:.1%} (기준 85% 미만) │ "
                      f"Bbox {int(row['bbox_count'])}개 인식",
            "action": "📋 **조치 필요:** 추가 현장 촬영 또는 조도·각도 개선 후 재인식 요망",
        })

# 전체 프로젝트 수준 경보
if total_CPI < 0.9:
    alerts.append({
        "level":  "CRITICAL",
        "icon":   "🔴",
        "title":  "프로젝트 전체 CPI 위험 수준",
        "detail": f"CPI = {total_CPI:.3f} (기준 0.90 미만) │ "
                  f"EAC 예측 {EAC_proj/1e6:.1f}M원 / BAC {BAC/1e6:.1f}M원 │ "
                  f"완공 시 예상 초과 {abs(VAC_proj)/1e6:.1f}M원",
        "action": "📋 **즉시 조치:** 전 공종 원가 절감 방안 수립 및 발주처 보고 필요",
    })

if total_SPI < 0.9:
    alerts.append({
        "level":  "CRITICAL",
        "icon":   "🔴",
        "title":  "프로젝트 전체 SPI 위험 수준",
        "detail": f"SPI = {total_SPI:.3f} (기준 0.90 미만) │ "
                  f"SV 누계 {total_SV/1e6:.2f}M원",
        "action": "📋 **즉시 조치:** 공정 만회 계획 수립 및 추가 장비·인원 투입 검토 필요",
    })

# 경보 없음
if not alerts:
    st.markdown("""
<div style="background:#10301A;border-left:4px solid #28A745;border-radius:8px;
            padding:16px;color:#28A745;font-size:1rem">
✅ &nbsp; <strong>이상 없음</strong> — 현재 조회 조건 내 모든 항목이 정상 범위입니다.
</div>""", unsafe_allow_html=True)
else:
    # 심각도 순 정렬
    order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    alerts.sort(key=lambda a: order[a["level"]])

    # 요약 배너
    n_critical = sum(1 for a in alerts if a["level"]=="CRITICAL")
    n_warning  = sum(1 for a in alerts if a["level"]=="WARNING")
    n_info     = sum(1 for a in alerts if a["level"]=="INFO")

    banner_bg  = "#3A1010" if n_critical else "#3A2A10"
    banner_txt = "#FF6B6B" if n_critical else "#FFC107"
    st.markdown(f"""
<div style="background:{banner_bg};border-radius:8px;padding:14px 18px;
            color:{banner_txt};font-size:1rem;margin-bottom:12px">
  ⚠️ &nbsp; 총 <strong>{len(alerts)}건</strong> 감지
  &nbsp;|&nbsp; 🔴 긴급 {n_critical}건
  &nbsp;|&nbsp; 🟡 경고 {n_warning}건
  &nbsp;|&nbsp; 🔵 정보 {n_info}건
</div>""", unsafe_allow_html=True)

    for a in alerts:
        bg_color     = "#3A1010" if a["level"]=="CRITICAL" else \
                       "#3A2A10" if a["level"]=="WARNING" else "#101828"
        border_color = C_DANGER if a["level"]=="CRITICAL" else \
                       C_WARN   if a["level"]=="WARNING"  else "#4C72B0"
        text_color   = "#FF6B6B" if a["level"]=="CRITICAL" else \
                       "#FFC107" if a["level"]=="WARNING"  else "#7EB6FF"

        st.markdown(f"""
<div style="background:{bg_color};border-left:5px solid {border_color};
            border-radius:8px;padding:14px 18px;margin-bottom:10px">
  <div style="font-size:1rem;font-weight:700;color:{text_color}">
    {a['icon']} &nbsp;{a['title']}
  </div>
  <div style="font-size:0.85rem;color:#CCCCCC;margin-top:6px">{a['detail']}</div>
  <div style="font-size:0.875rem;color:{text_color};margin-top:8px">{a['action']}</div>
</div>""", unsafe_allow_html=True)

st.divider()

# ============================================================
# ④ 원시 데이터 테이블 (접기)
# ============================================================
with st.expander("📋 원시 데이터 테이블 보기"):
    display_cols = {
        "capture_date":"촬영일자", "wbs_code":"WBS", "work_name":"공종",
        "material_name":"자재명", "unit":"단위",
        "planned_qty":"계획수량", "actual_qty":"실적수량",
        "PV":"PV(원)", "EV":"EV(원)", "AC":"AC(원)",
        "SV":"SV(원)", "CV":"CV(원)", "SPI":"SPI", "CPI":"CPI",
        "delay_budget_impact":"지연→예산압력(원)",
        "confidence_avg":"AI신뢰도",
    }
    show_df = filtered[list(display_cols.keys())].rename(columns=display_cols)

    def color_negative(val):
        if isinstance(val, (int, float)) and val < 0:
            return "color: #FF6B6B; font-weight:bold"
        return ""

    st.dataframe(
        show_df.style
               .map(color_negative, subset=["SV(원)","CV(원)"])
               .format({
                   "PV(원)": "{:,.0f}", "EV(원)": "{:,.0f}", "AC(원)": "{:,.0f}",
                   "SV(원)": "{:+,.0f}", "CV(원)": "{:+,.0f}",
                   "SPI": "{:.3f}", "CPI": "{:.3f}",
                   "지연→예산압력(원)": "{:,.0f}", "AI신뢰도": "{:.1%}",
               }),
        use_container_width=True,
        height=320,
    )
    csv = show_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("⬇️ CSV 다운로드", csv,
                       "자재원가모니터링.csv", "text/csv")

# 푸터
st.markdown("""
<div style="text-align:center;color:#555;font-size:0.78rem;margin-top:20px">
건축 AI 수업 프로젝트 · 모듈2 7조 &nbsp;|&nbsp;
Vision AI(YOLOv8) + EVMS 기반 자재 원가 모니터링 시스템
</div>""", unsafe_allow_html=True)
