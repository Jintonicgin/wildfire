# 📄 산불 확산 예측 머신러닝 결과 보고서

## 📚 보고서 구성

이 디렉토리는 산불 확산 예측 머신러닝 프로젝트의 종합적인 결과 보고서를 포함합니다.

### 📋 파일 구성

#### 📖 주요 보고서
- **`wildfire_ml_results_report.md`** - 📊 **메인 결과 보고서**
  - 프로젝트 개요 및 목표
  - 데이터셋 구성 및 특징
  - 모델 아키텍처 (3개 예측 모델)
  - 성능 결과 및 평가 지표
  - 시각화 결과 분석
  - 실무 적용 가능성 및 제한사항

- **`technical_details.md`** - 🔬 **기술 상세 문서**
  - 프로젝트 구조 분석
  - 앙상블 모델 구현 상세
  - 피처 엔지니어링 파이프라인
  - 실시간 예측 시스템 구조
  - 성능 최적화 전략

#### 🖼️ 시각화 자료
- **`actual_vs_predicted_damage_area.png`** - 실제 vs 예측 피해 면적 산점도
- **`feature_importance_v2.png`** - XGBoost 피처 중요도 차트
- **`direction_model_confusion_matrix.png`** - 방향 예측 모델 혼동 행렬
- **`speed_model_confusion_matrix.png`** - 속도 예측 모델 혼동 행렬
- **`training_data_target_distribution.png`** - 훈련 데이터 타겟 분포

## 🎯 핵심 성과 요약

### 📈 모델 성능
- **확산 속도 분류**: 84.2% 정확도 (5개 모델 앙상블)
- **피해 면적 예측**: MAE 50-80ha, 소규모 화재 15-25ha
- **확산 방향 분류**: 8방향 분류, 75-80% 정확도

### 🚀 기술적 혁신
- **실시간 예측**: Google Earth Engine + NASA POWER API 연동
- **앙상블 학습**: RF + XGBoost + LightGBM + CatBoost + GB 조합
- **200+ 피처**: 자동 피처 엔지니어링 파이프라인
- **Flask 웹앱**: 사용자 친화적 예측 인터페이스

## 📖 읽는 법

### 🔰 입문자용
1. **`wildfire_ml_results_report.md`**의 "프로젝트 개요" 부터 시작
2. 시각화 이미지들을 통해 결과 이해
3. "실무 적용 가능성" 섹션으로 마무리

### 🔬 기술자용  
1. **`technical_details.md`**에서 구현 상세 확인
2. 코드 구조 및 알고리즘 분석
3. 성능 최적화 전략 검토

### 📊 의사결정자용
1. "핵심 성과 요약" 먼저 확인
2. "모델 성능 결과" 섹션의 지표 검토  
3. "결론 및 시사점"의 비즈니스 임팩트 분석

## 🔗 관련 자료

### 📁 소스 코드 위치
```
../wildfire/wildfire/dataset/
├── train.py              # 모델 학습
├── predict.py            # 실시간 예측
├── wild_fire_ml.py       # 개별 모델 학습
└── *.joblib              # 학습된 모델 파일
```

### 🌐 웹 애플리케이션
```
../wildfire/
├── run.py                # Flask 서버 실행
├── wildfire/views/       # 웹 인터페이스
└── wildfire/templates/   # HTML 템플릿
```

## 📞 문의사항

프로젝트 관련 문의사항이나 기술적 질문은 다음을 통해 연락해주세요:

- **프로젝트**: WildFire_projects
- **리포지토리**: wildfire/
- **작성일**: 2025년 1월

---

**🎉 이 보고서가 산불 예방 및 대응 연구에 도움이 되기를 바랍니다!**