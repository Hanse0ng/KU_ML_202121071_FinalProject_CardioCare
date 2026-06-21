### KU_ML_202121071_FinalProject_CardioCare

# 1. 개요
- 본 레포지토리는 건국대학교 글로컬캠퍼스 2026년 1학기 기계학습(8253) 기말 프로젝트입니다.
- CardioCare(임상 데이터로부터 심장병 발병 가능성을 예측하여 심장 전문의의 의사결정을 지원하는 시스템)를 주제로 종단간 머신러닝 시스템을 구축하는 것을 목표로 합니다.

# 2. 재현
- 채점자는 다음 과정을 통해 전 과정을 재현할 수 있습니다.
1. 저장소 clone
2. 의존성 설치: pip install -r requirements.txt
3. 학습 실행: python src/train.py
4. docker build
5. python -m unittest