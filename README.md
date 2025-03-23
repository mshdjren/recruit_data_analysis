# DS-Projcet (Ajou University for sensor-big data course)
**2014~2018년도경기도 대학 졸업생 취업 현황 데이터 분석 프로젝트**

## 프로젝트 개요
본 프로젝트는 경기도 데이터센터에서 제공하는 2014~2018년 경기도 소재 대학 졸업생들의 취업 및 진로 현황 데이터를 분석하여 학과별 진로 방향의 유사성을 확인하고, 연도별 변화 및 경향을 파악하는 것을 목표로 합니다. 이를 통해 졸업생들의 진로 선택에 대한 인사이트를 도출하고, 향후 진로 계획에 도움을 주고자 합니다.

## 데이터 개요
- **출처**: 경기데이터드림
- **데이터 형식**: CSV 파일
- **분석 대상**: 경기도 소재 여러 대학(아주대학교 포함)의 2014~2018년 졸업생 데이터
- **주요 컬럼**:
  - 기준년도, 학교명, 단과대학명, 학과명(전공), 졸업자수, 국내+해외취업자수, 진학자수, 기타인원수1(입대자 등), 기타인원수2, 취업률(%)
  - 
<img src=https://github.com/mshdjren/recruit_data_analysis/blob/master/data.jpg>

## 분석 방법론

### 1. 데이터 전처리
- 원본 데이터에서 취업자 수가 0에 가까운 레이블을 제거하고 주요 레이블(취업자수, 진학자수 등) 중심으로 데이터를 재구성.

### 2. 알고리즘 선택
- **K-Means Clustering**:
  - 학과별 진로 방향(취업, 대학원 진학 등)의 유사성을 군집화.
  - 클러스터 내 단과대학의 일치도를 확인하여 군집화의 적절성을 평가.

### 3. 분석 절차
- 연도별 클러스터링 결과 비교 (2014~2018).
- 전체 연도 데이터를 활용한 클러스터링 진행.
- 클러스터 개수(k)를 단계적으로 증가시켜 결과를 분석 (k=2, k=5, k=9 등).
- 단과대학 수와 k값을 유사하게 설정하여 진로 방향의 유사성을 평가.

### 4. 시각화
- 학과를 x축으로, 클러스터링 결과의 특징 합계를 y축으로 하는 산점도 그래프 생성.
- 연도별 및 전체 데이터를 기반으로 클러스터링 결과 시각화.

## 주요 결과 및 한계

### 결과
- 단과대학이 유사할수록 진로 방향이 비슷할 가능성이 높음을 확인.
- k=9 (단과대학 수와 동일)일 때 가장 적절한 클러스터링 결과를 도출.

### 한계
1. 데이터에서 사용된 레이블(취업자수, 진학자수 등)이 진로 방향을 충분히 구체적으로 반영하지 못함.
2. Features 간 중요도를 고려하지 못해 현실적인 진로 분석에는 제한적임.
<img src=https://github.com/mshdjren/recruit_data_analysis/blob/master/k%3D9.jpg>
<img src=https://github.com/mshdjren/recruit_data_analysis/blob/master/k%3D20.jpg>
