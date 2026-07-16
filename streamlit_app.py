from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Pima Diabetes EDA",
    page_icon="🩺",
    layout="wide",
)

DATA_PATH = Path(__file__).parent / "data" / "diabetes.csv"
ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
FEATURE_KO = {
    "Pregnancies": "임신 횟수",
    "Glucose": "혈당",
    "BloodPressure": "이완기 혈압",
    "SkinThickness": "피부 두께",
    "Insulin": "인슐린",
    "BMI": "체질량지수(BMI)",
    "DiabetesPedigreeFunction": "가족력 점수",
    "Age": "나이",
}

available_fonts = {font.name for font in font_manager.fontManager.ttflist}
chart_font = next(
    (name for name in ["Malgun Gothic", "NanumGothic", "AppleGothic"] if name in available_fonts),
    "DejaVu Sans",
)
sns.set_theme(style="whitegrid", font=chart_font)
plt.rcParams["font.family"] = chart_font
plt.rcParams["axes.unicode_minus"] = False


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    expected = set(FEATURES + ["Outcome"])
    missing = expected.difference(data.columns)
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {sorted(missing)}")
    return data


@st.cache_data
def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hidden = pd.DataFrame(
        {
            "개수": raw[ZERO_COLS].eq(0).sum(),
            "비율(%)": raw[ZERO_COLS].eq(0).mean().mul(100),
        }
    ).sort_values("비율(%)", ascending=False)

    cleaned = raw.copy()
    cleaned[ZERO_COLS] = cleaned[ZERO_COLS].replace(0, np.nan)
    cleaned[ZERO_COLS] = cleaned[ZERO_COLS].fillna(cleaned[ZERO_COLS].median())
    return cleaned, hidden


@st.cache_resource
def train_model(raw: pd.DataFrame):
    x = raw[FEATURES].copy()
    y = raw["Outcome"].copy()
    x[ZERO_COLS] = x[ZERO_COLS].replace(0, np.nan)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    probability = pipeline.predict_proba(x_test)[:, 1]
    return pipeline, x_test, y_test, probability


def labeled_outcome(series: pd.Series) -> pd.Series:
    return series.map({0: "음성", 1: "양성"})


try:
    df_raw = load_data(DATA_PATH)
except (FileNotFoundError, ValueError) as error:
    st.error(f"데이터를 불러오지 못했습니다: {error}")
    st.stop()
    raise SystemExit(1)  # 일반 Python 실행으로 점검할 때도 후속 코드 실행 방지

df_clean, hidden_summary = clean_data(df_raw)

st.title("🩺 Pima Indians Diabetes 시각화 EDA 대시보드")
st.caption(
    "768명의 임상 측정값에서 숨은 결측, 그룹 차이, 변수 관계와 분류 임계값을 탐색합니다."
)

with st.sidebar:
    st.header("필터")
    selected_outcomes = st.multiselect(
        "진단 결과",
        options=[0, 1],
        default=[0, 1],
        format_func=lambda value: "음성(0)" if value == 0 else "양성(1)",
    )
    age_range = st.slider(
        "나이 범위",
        int(df_clean["Age"].min()),
        int(df_clean["Age"].max()),
        (int(df_clean["Age"].min()), int(df_clean["Age"].max())),
    )
    glucose_range = st.slider(
        "혈당 범위",
        int(df_clean["Glucose"].min()),
        int(df_clean["Glucose"].max()),
        (int(df_clean["Glucose"].min()), int(df_clean["Glucose"].max())),
    )
    st.divider()
    st.info("필터는 EDA 차트와 데이터 표에 적용됩니다. 모델 평가는 전체 데이터로 고정됩니다.")

if not selected_outcomes:
    st.warning("진단 결과를 하나 이상 선택하세요.")
    st.stop()

filtered = df_clean[
    df_clean["Outcome"].isin(selected_outcomes)
    & df_clean["Age"].between(*age_range)
    & df_clean["Glucose"].between(*glucose_range)
].copy()

if filtered.empty:
    st.warning("현재 조건에 해당하는 데이터가 없습니다. 필터 범위를 넓혀주세요.")
    st.stop()

positive_rate = filtered["Outcome"].mean() * 100
metric_cols = st.columns(4)
metric_cols[0].metric("선택 표본", f"{len(filtered):,}명", f"전체 {len(df_raw):,}명")
metric_cols[1].metric("양성 비율", f"{positive_rate:.1f}%")
metric_cols[2].metric("중앙 혈당", f"{filtered['Glucose'].median():.1f}")
metric_cols[3].metric("중앙 BMI", f"{filtered['BMI'].median():.1f}")

tab_overview, tab_relation, tab_quality, tab_model, tab_data = st.tabs(
    ["📊 개요", "🔗 변수 관계", "🧹 데이터 품질", "🤖 모델 평가", "📋 데이터"]
)

with tab_overview:
    left, right = st.columns([0.8, 1.2])
    with left:
        st.subheader("진단 결과 분포")
        plot_df = filtered.assign(진단결과=labeled_outcome(filtered["Outcome"]))
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.countplot(
            data=plot_df,
            x="진단결과",
            hue="진단결과",
            order=[value for value in ["음성", "양성"] if value in plot_df["진단결과"].unique()],
            palette={"음성": "#4C78A8", "양성": "#E45756"},
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel("인원")
        for container in ax.containers:
            ax.bar_label(container, padding=3)
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("변수별 분포")
        feature = st.selectbox(
            "확인할 변수",
            FEATURES,
            index=FEATURES.index("Glucose"),
            format_func=lambda value: f"{FEATURE_KO[value]} ({value})",
        )
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        sns.histplot(
            data=plot_df,
            x=feature,
            hue="진단결과",
            hue_order=["음성", "양성"],
            kde=True,
            element="step",
            common_norm=False,
            ax=axes[0],
        )
        sns.boxplot(
            data=plot_df,
            x="진단결과",
            y=feature,
            order=[value for value in ["음성", "양성"] if value in plot_df["진단결과"].unique()],
            hue="진단결과",
            palette={"음성": "#4C78A8", "양성": "#E45756"},
            legend=False,
            ax=axes[1],
        )
        axes[0].set_title(f"{FEATURE_KO[feature]} 분포")
        axes[1].set_title("진단 결과별 비교")
        axes[1].set_xlabel("")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with tab_relation:
    st.subheader("상관관계 히트맵")
    corr = filtered[FEATURES + ["Outcome"]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.4,
        ax=ax,
    )
    st.pyplot(fig)
    plt.close(fig)
    st.caption("상관계수는 선형 관계의 크기이며, 인과관계를 의미하지 않습니다.")

    st.subheader("두 변수 산점도")
    scatter_cols = st.columns(2)
    x_feature = scatter_cols[0].selectbox(
        "X축",
        FEATURES,
        index=FEATURES.index("Glucose"),
        format_func=lambda value: f"{FEATURE_KO[value]} ({value})",
    )
    y_feature = scatter_cols[1].selectbox(
        "Y축",
        FEATURES,
        index=FEATURES.index("BMI"),
        format_func=lambda value: f"{FEATURE_KO[value]} ({value})",
    )
    scatter_df = filtered.assign(진단결과=labeled_outcome(filtered["Outcome"]))
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        data=scatter_df,
        x=x_feature,
        y=y_feature,
        hue="진단결과",
        hue_order=["음성", "양성"],
        palette={"음성": "#4C78A8", "양성": "#E45756"},
        alpha=0.7,
        ax=ax,
    )
    st.pyplot(fig)
    plt.close(fig)

with tab_quality:
    st.subheader("0으로 위장된 숨은 결측")
    st.write(
        "혈당·혈압·피부 두께·인슐린·BMI의 0은 실제 측정값으로 부자연스러워 "
        "결측으로 간주하고 중앙값으로 대체했습니다."
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    chart_data = hidden_summary.sort_values("비율(%)")
    bars = ax.barh(
        chart_data.index,
        chart_data["비율(%)"],
        color=sns.color_palette("Reds", n_colors=len(chart_data)),
    )
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in chart_data["비율(%)"]], padding=3)
    ax.set_xlabel("0으로 기록된 비율(%)")
    ax.set_xlim(0, chart_data["비율(%)"].max() * 1.15)
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(hidden_summary.round(2), width="stretch")

    compare_feature = st.selectbox(
        "처리 전후를 비교할 변수",
        ZERO_COLS,
        index=ZERO_COLS.index("Insulin"),
        format_func=lambda value: f"{FEATURE_KO[value]} ({value})",
    )
    compare = pd.DataFrame(
        {
            "원본(0 포함)": df_raw[compare_feature],
            "정제 후": df_clean[compare_feature],
        }
    ).melt(var_name="상태", value_name=compare_feature)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.histplot(
        data=compare,
        x=compare_feature,
        hue="상태",
        element="step",
        common_norm=False,
        bins=30,
        ax=ax,
    )
    ax.set_title(f"{FEATURE_KO[compare_feature]} 결측 처리 전후 분포")
    st.pyplot(fig)
    plt.close(fig)

with tab_model:
    st.subheader("로지스틱 회귀: 임계값에 따른 진단 성능")
    st.info(
        "EDA를 예측 모델로 확장한 교육용 결과입니다. 실제 의료 진단이나 치료 결정에 사용할 수 없습니다."
    )
    model, x_test, y_test, probability = train_model(df_raw)
    threshold = st.slider("양성 판정 임계값", 0.10, 0.90, 0.50, 0.05)
    prediction = (probability >= threshold).astype(int)

    model_cols = st.columns(6)
    model_cols[0].metric("Accuracy", f"{accuracy_score(y_test, prediction):.3f}")
    model_cols[1].metric("Precision", f"{precision_score(y_test, prediction, zero_division=0):.3f}")
    model_cols[2].metric("Recall", f"{recall_score(y_test, prediction, zero_division=0):.3f}")
    model_cols[3].metric("F1", f"{f1_score(y_test, prediction, zero_division=0):.3f}")
    model_cols[4].metric("ROC-AUC", f"{roc_auc_score(y_test, probability):.3f}")
    model_cols[5].metric("PR-AUC", f"{average_precision_score(y_test, probability):.3f}")

    left, right = st.columns([0.8, 1.2])
    with left:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(y_test, prediction),
            display_labels=["음성", "양성"],
        ).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"혼동행렬 (임계값 {threshold:.2f})")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        coefficient = model.named_steps["logistic"].coef_[0]
        coef_df = pd.DataFrame({"변수": FEATURES, "표준화 계수": coefficient}).sort_values("표준화 계수")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = ["#4C78A8" if value < 0 else "#E45756" for value in coef_df["표준화 계수"]]
        ax.barh(coef_df["변수"], coef_df["표준화 계수"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("로지스틱 회귀 표준화 계수")
        st.pyplot(fig)
        plt.close(fig)
    st.caption(
        "임계값을 낮추면 일반적으로 Recall이 올라가지만 False Positive도 증가합니다. "
        "의료 선별에서는 놓친 양성(False Negative)의 비용을 함께 고려해야 합니다."
    )

with tab_data:
    st.subheader("필터 적용 데이터")
    display_df = filtered.copy()
    display_df.insert(0, "Diagnosis", labeled_outcome(display_df["Outcome"]))
    st.dataframe(display_df, width="stretch", hide_index=True)
    st.download_button(
        "CSV 다운로드",
        data=filtered.to_csv(index=False).encode("utf-8-sig"),
        file_name="pima_diabetes_filtered.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "데이터 출처: UCI / Kaggle Pima Indians Diabetes Database · "
    "본 대시보드는 학습용이며 의료 조언을 제공하지 않습니다."
)
