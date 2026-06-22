# KU_ML_202121071_FinalProject_CardioCare

## 1. 개요
- 본 레포지토리는 건국대학교 글로컬캠퍼스 2026년 1학기 기계학습(8253) 기말 프로젝트입니다.
- CardioCare(임상 데이터로부터 심장병 발병 가능성을 예측하여 심장 전문의의 의사결정을 지원하는 시스템)를 주제로 종단간 머신러닝 시스템을 구축하는 것을 목표로 합니다.

## 2. 재현
- 채점자는 다음 과정을 통해 전 과정을 재현할 수 있습니다.
1. 저장소 clone
2. 의존성 설치: pip install -r requirements.txt
3. 학습 실행:
   - python src/train.py
   - 또는
   - py src/train.py
4. 도커 빌드 및 실행:
   - 빌드: docker build -t cardiocare:1.0 .
   - 실행: docker run cardiocare:1.0
5. 단위 테스트:
   - python -m unittest
   - 또는
   - py -m unittest

## 3. 모니터링
- 채점자는 "2. 재현" 과정 이후 다음 명령어를 통해 과제 요구사항 5번(모니터링 및 데이터 드리프트 탐지)의 결과를 확인할 수 있습니다.
   - python src/monitor.py
   - 또는
   - py src/monitor.py

## 4. 기타
- 데이터셋: 'data/'에는 기본적으로 사용한 Cleveland 부분 집합 데이터셋인 'heart_disease.csv'가 존재하며, 없더라도 'train.py' 실행 초기에 'data.py'를 통해 데이터가 로드됩니다.