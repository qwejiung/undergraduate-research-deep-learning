# 학부연구생 연구 포트폴리오: Cross-Cancer Transfer Learning

## 소개
본 레포지토리는 **학부 연구생으로 참여한 딥러닝 기반 의료 AI 연구 경험**을 바탕으로,  
연구 과정에서 수행한 핵심 코드와 실험 구조를 **개인 포트폴리오를 위해 일부만 공개한 저장소**입니다.

## 연구 배경
본 연구는 다음 논문을 기반으로 합니다.

> **Cross-Cancer Transfer Learning for Gastric Cancer Risk Prediction from Electronic Health Records**  
> Diagnostics, 2025  
> 저자: Hong, D.; **Kim, Jiung**; Jung, J.

본 연구는 **구조화된 전자의무기록(EHR) 데이터**를 활용하여,  
서로 다른 소화기계 암에서 공통적으로 나타나는 임상 신호를 전이학습을 통해 학습하고  
이를 **위암(Gastric Cancer) 위험 예측**에 활용할 수 있는지를 분석합니다.


## 연구 개요
- **주제**: 구조화 EHR 기반 위암 위험 예측
- **방법**: 다수 소화기계 암을 활용한 전이학습 (Transfer Learning)
- **모델**: MLP 기반 딥러닝 모델

## 문제 정의
위암은 발생 빈도가 낮아 학습 가능한 데이터가 제한적이며,  
구조화된 EHR 기반 예측 모델은 라벨 부족과 클래스 불균형으로 성능 저하가 발생한다.  
본 연구는 다른 소화기계 암에서 학습한 정보를 위암 예측에 전이할 수 있는지를 분석한다.

## 방법론 요약
대장암, 식도암, 간암, 췌장암 데이터를 활용해  
MLP 기반 딥러닝 모델을 사전 학습(pretraining)한 뒤  
위암 데이터에 미세 조정(fine-tuning)하는 전이학습 방식을 적용하였다.

## 비교 모델
- Logistic Regression
- XGBoost
- Scratch MLP (사전학습 x)

## 주요 결과
전이학습 모델은 AUROC와 F1-score에서 비전이 모델 대비 일관된 성능 향상을 보였다.  
특히 위암 학습 데이터가 적을수록 성능 안정성과 우수성이 더욱 두드러졌다.