# Pima Diabetes EDA Dashboard

Pima Indians Diabetes 데이터의 숨은 결측, 진단 그룹별 분포, 변수 관계와 로지스틱 회귀 임계값을 탐색하는 Streamlit 대시보드입니다.

> 교육용 분석이며 실제 의료 진단이나 치료 결정에 사용할 수 없습니다.

## 프로젝트 구조

```text
pima-diabetes-dashboard/
├── data/
│   └── diabetes.csv
├── streamlit_app.py
├── requirements.txt
├── packages.txt
├── .gitignore
└── README.md
```

## 로컬 실행

가장 쉬운 방법은 VS Code에서 `pima-diabetes-dashboard` 폴더 자체를 연 다음
**Terminal → New Terminal**을 선택하는 것입니다. 그러면 경로를 길게 입력하지 않아도 됩니다.

터미널 프롬프트가 프로젝트 폴더를 가리키는지 확인합니다.

```powershell
Get-Location
dir
```

`dir` 결과에 `streamlit_app.py`, `requirements.txt`, `data` 폴더가 보이면 아래를 실행합니다.

```powershell
cd "C:\Users\82102\Desktop\강의_AI_Pair\pima-diabetes-dashboard"
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

브라우저가 자동으로 열리지 않으면 <http://localhost:8501>에 접속합니다.

### 다음부터 다시 실행할 때

패키지 설치는 최초 한 번만 필요합니다. 다음 실행부터는 프로젝트 폴더에서 한 줄만 입력합니다.

```powershell
python -m streamlit run streamlit_app.py
```

종료는 앱을 실행한 터미널에서 `Ctrl+C`를 누릅니다.

### 자주 발생하는 오류

- `File does not exist`: `dir`로 현재 폴더에 `streamlit_app.py`가 있는지 확인합니다.
- `streamlit is not recognized`: `streamlit run` 대신 `python -m streamlit run`을 사용합니다.
- `ModuleNotFoundError`: `python -m pip install -r requirements.txt`를 다시 실행합니다.
- 포트가 사용 중이면 `python -m streamlit run streamlit_app.py --server.port 8502`로 실행합니다.

## ST2에서 가져온 설계 패턴

- `@st.cache_data`: CSV 로드·정제 결과를 재사용합니다.
- `@st.cache_resource`: 학습된 로지스틱 회귀 모델을 재사용합니다.
- 사이드바: 나이·혈당·Outcome 필터를 본문과 분리합니다.
- 탭: 개요·변수 관계·데이터 품질·모델·원본 표를 목적별로 나눕니다.
- `st.metric`: 표본 수, 양성 비율, 혈당·BMI 요약을 한눈에 보여줍니다.
- `plt.close(fig)`: rerun 때 이전 Figure가 메모리에 계속 쌓이지 않게 합니다.

## GitHub 업로드

처음 한 번만 Git 작성자 정보를 설정합니다. GitHub 이메일을 공개하고 싶지 않다면 GitHub의 `noreply` 이메일을 사용하세요.

```powershell
git config --global user.name "YOUR_GITHUB_NAME"
git config --global user.email "YOUR_EMAIL_OR_NOREPLY_EMAIL"
```

현재 PC에 GitHub CLI(`gh`)가 없다면 GitHub 웹사이트에서 `pima-diabetes-dashboard`라는 **빈 저장소**를 먼저 만들고 아래의 "GitHub 웹에서 빈 저장소를 먼저 생성했다면" 절을 사용하세요.

GitHub CLI를 사용하는 경우:

```powershell
git init
git add .
git commit -m "Build Pima Diabetes EDA dashboard"
git branch -M main
gh auth login
gh repo create pima-diabetes-dashboard --public --source=. --remote=origin --push
```

GitHub 웹에서 빈 저장소를 먼저 생성했다면:

```powershell
git init
git add .
git commit -m "Build Pima Diabetes EDA dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/pima-diabetes-dashboard.git
git push -u origin main
```

## Streamlit Community Cloud 배포

1. <https://share.streamlit.io>에 GitHub 계정으로 로그인합니다.
2. **Create app**을 선택합니다.
3. GitHub 저장소와 `main` 브랜치를 선택합니다.
4. Main file path에 `streamlit_app.py`를 입력합니다.
5. 원하는 App URL을 정하고 **Deploy**를 누릅니다.

GitHub에 새 커밋을 push하면 배포된 앱도 자동으로 업데이트됩니다.

## 혼자 확장하는 방법

1. 질문을 먼저 적습니다: 예) “나이 구간별 양성 비율은 다른가?”
2. 필요한 필터 하나와 차트 하나만 추가합니다.
3. 차트 바로 아래에 과장하지 않은 결론 한 줄을 둡니다.
4. 새 패키지를 import했다면 `requirements.txt`에도 추가합니다.
5. 로컬에서 확인한 뒤 `git add` → `git commit` → `git push` 순서로 배포를 갱신합니다.
