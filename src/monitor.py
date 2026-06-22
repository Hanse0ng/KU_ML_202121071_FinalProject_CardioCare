import os
import sys
import json
import yaml
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import mlflow

from preprocessing import IQRClamp, clean_data
from inference import get_best_model_artifacts
from train import SEED

# 로깅 기반 계측 설정
logger = logging.getLogger("inference_logger")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("data/inference.log", encoding="utf-8")
formatter = logging.Formatter('%(asctime)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

# 추론 경로 로깅: 타임스탬프, 모델 버전, 입력 shape, 예측값, 실제 정답
def log_inference(model_version, input_shape, predictions, y_true=None):
    
    log_msg = f"Model: {model_version} | Input Shape: {input_shape} | Preds: {predictions.tolist()}"
    if y_true is not None:
        log_msg += f" | Ground Truth: {y_true.tolist()}"
    logger.info(log_msg)

def main():
    # 데이터 로드 및 전처리
    data_path = 'data/heart_disease.csv'
    if not os.path.exists(data_path):
        print(f"데이터셋('{data_path}')을 찾을 수 없습니다.")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    df_cleaned = clean_data(df)
    
    X = df_cleaned.drop(columns=['target'])
    y = df_cleaned['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )
    
    # 모델 및 아티팩트 불러오기
    try:
        model_dir, bounds_path = get_best_model_artifacts()
        with open(bounds_path, "r", encoding="utf-8") as f:
            train_bounds = json.load(f)
            
        # 모델의 버전(Run ID) 추출
        with open(os.path.join(model_dir, "MLmodel"), "r", encoding="utf-8") as f:
            model_version = yaml.safe_load(f).get("run_id", "Unknown")
            
        model = mlflow.pyfunc.load_model(model_dir)
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        sys.exit(1)
        
    # --- 원본 테스트 셋 추론 및 로깅 ---
    X_test_clamped = IQRClamp(X_test, train_bounds)
    
    preds_original = model.predict(X_test_clamped)
    log_inference(model_version, X_test_clamped.shape, preds_original, y_test.values)
    
    acc_original = balanced_accuracy_score(y_test, preds_original)
    
    # 테스트 셋 복사본 인위적 이동
    X_test_drifted = X_test.copy()
    np.random.seed(SEED)
    
    # 확실한 결과 확인을 위해 4개의 연속형 특성들을 이동
    # 분산 증가를 위해 np.random 적용
    # age -30, thalach -50, oldpeak +3, trestbps +40
    X_test_drifted['age'] = X_test_drifted['age'] - np.random.normal(loc=20, scale=5, size=len(X_test_drifted))
    X_test_drifted['thalach'] = X_test_drifted['thalach'] - np.random.normal(loc=50, scale=10, size=len(X_test_drifted))
    X_test_drifted['oldpeak'] = X_test_drifted['oldpeak'] + np.random.normal(loc=3.0, scale=1.0, size=len(X_test_drifted))
    X_test_drifted['trestbps'] = X_test_drifted['trestbps'] + np.random.normal(loc=40, scale=10, size=len(X_test_drifted))

    # KS 2-Samp 테스트 및 플래그
    print("\n[Data Drift K-S Test 결과]")
    continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    drift_flags = []
    
    for feature in continuous_features:
        # Train의 원본 vs 테스트의 드리프트된 데이터
        train_data = X_train[feature].dropna()
        test_data = X_test_drifted[feature].dropna()
        
        stat, p_value = ks_2samp(train_data, test_data)
        
        is_drifted = p_value < 0.05
        if is_drifted:
            drift_flags.append(feature)
            flag_str = "[DRIFT DETECTED]"
        else:
            flag_str = "[OK]"
            
        print(f"Feature: {feature:10s} | p-value: {p_value:.4f} | {flag_str}")
        
    # --- 4. 드리프트된 데이터 예측 및 성능 비교 ---
    # 드리프트된 데이터도 IQR 전처리만 거쳐 추론 (결측치는 파이프라인이 자동 처리)
    X_test_drifted_clamped = IQRClamp(X_test_drifted, train_bounds)
    
    preds_drifted = model.predict(X_test_drifted_clamped)
    log_inference(model_version + "_drifted", X_test_drifted_clamped.shape, preds_drifted, y_test.values)
    
    acc_drifted = balanced_accuracy_score(y_test, preds_drifted)
    
    print("\n[모델 성능 변화 (Balanced Accuracy)]")
    print(f"원본 테스트셋 성능  : {acc_original:.4f}")
    print(f"드리프트 테스트셋 성능: {acc_drifted:.4f}")
    print(f"성능 하락 폭        : {acc_original - acc_drifted:.4f}\n")
    
    # 시간에 따른 지표 변화 시계열 그래프
    # 30일 간의 합성 타임스탬프에 맞춰 심해지는 드리프트를 실제 데이터에 적용하고 평가
    dates = [datetime.date.today() - datetime.timedelta(days=i) for i in range(30, 0, -1)]
    accuracies = []

    for i in range(30):
        # 1~15일: 원본 데이터 상태 유지 (정상 데이터)
        if i < 15:
            X_temp = X_test.copy()
            X_temp_clamped = IQRClamp(X_temp, train_bounds)
            
            acc = balanced_accuracy_score(y_test, model.predict(X_temp_clamped))
            accuracies.append(acc)
            
        # 16~30일: 드리프트 강도를 0%에서 100%까지 증가
        else:
            intensity = (i - 14) / 15.0  # 날짜가 지날수록 1.0(최대)에 가까워짐
            
            X_temp = X_test.copy()
            # 평균 이동 및 분산 증가를 위해 난수 적용
            X_temp['age'] -= np.random.normal(20 * intensity, 5, len(X_temp))
            X_temp['thalach'] -= np.random.normal(50 * intensity, 10, len(X_temp))
            X_temp['oldpeak'] += np.random.normal(3.0 * intensity, 1.0, len(X_temp))
            X_temp['trestbps'] += np.random.normal(40 * intensity, 10, len(X_temp))
            
            # 클램핑 후 실제 예측 수행
            X_temp_clamped = IQRClamp(X_temp, train_bounds)
            acc = balanced_accuracy_score(y_test, model.predict(X_temp_clamped))
            accuracies.append(acc)
    
    plt.figure(figsize=(10, 5))
    plt.plot(dates, accuracies, marker='o', linestyle='-', color='b', label='Balanced Accuracy')
    plt.axhline(y=acc_original, color='g', linestyle='--', label='Baseline Performance')
    plt.axvline(x=dates[15], color='r', linestyle=':', label='Drift Inject Point')
    
    plt.title('Model Performance Over Time (Synthetic Timestamps)')
    plt.xlabel('Date')
    plt.ylabel('Balanced Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 그래프를 팝업으로 띄우기 (작성자 요청)
    plt.show()

if __name__ == "__main__":
    main()
