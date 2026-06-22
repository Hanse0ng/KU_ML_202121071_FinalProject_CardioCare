# 모델 학습을 담당하는 메인 스크립트
# 총 3개의 모델을 학습/비교 (Logistic Regression, SVC, Random Forest)

from preprocessing import clean_data, IQRClamp
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_validate, GridSearchCV
import os
import mlflow
import mlflow.sklearn

# mlflow의 mlruns/ 사용 제한 해제
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# 올바른 재현을 위한 시드 고정 및 train/test 분할 비율 명시
# 재현을 위해 시드(SEED)는 어떤 값이든 하나의 값을 고정 사용하면 되므로 1로 설정
# 비율(test_size)는 8(train):2(test) 비율로 설정
# 학습에 충분히 많은 데이터를 할당하면서 성능을 객관적으로 평가하기에 충분한 테스트 데이터를 확보하는 적절한 비율이기 때문
SEED = 1
test_size = 0.2 # 테스트 20%, 훈련 80%

import sys
import subprocess

if __name__ == "__main__":
    # 데이터 읽어오기 (없으면 자동 다운로드)
    data_path = 'data/heart_disease.csv'
    if not os.path.exists(data_path):
        print(f"'{data_path}'를 찾을 수 없습니다. data/data.py를 실행하여 다운로드합니다.")
        try:
            subprocess.run([sys.executable, "data/data.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"데이터 다운로드 스크립트 실행 실패: {e}")
            sys.exit(1)
            
    heart_disease_df = pd.read_csv(data_path)

    # 데이터 기초 정리(데이터 shape 변경 작업은 분할 전에 수행)
    df_cleaned = clean_data(heart_disease_df)

    # 독립 변수(X)와 종속 변수(y) 분리
    X = df_cleaned.drop(columns=['target'])
    y = df_cleaned['target']

    # train/test 데이터 분할 (데이터 누수 방지)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=SEED,
        stratify=y
    )

    # 이상치 처리
    X_train_clamped, train_bounds = IQRClamp(X_train)
    X_test_clamped = IQRClamp(X_test, train_bounds)

    # 최종 데이터
    X_train_final = X_train_clamped
    X_test_final = X_test_clamped

    # 세 모델에 대한 파이프라인 구성 (KNNImputer를 파이프라인 안으로 삽입)
    from sklearn.impute import KNNImputer
    
    pipelines = {
        "LogisticRegression": Pipeline(steps=[
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler()),
            ("feature_selection", SelectFromModel(RandomForestClassifier(random_state=SEED))),
            ("model", LogisticRegression(random_state=SEED))
        ]),
        "SVC": Pipeline(steps=[
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler()),
            ("feature_selection", SelectFromModel(RandomForestClassifier(random_state=SEED))),
            ("model", SVC(random_state=SEED))
        ]),
        # RandomForest는 거리 기반 모델이 아니므로 스케일러 생략
        "RandomForest": Pipeline(steps=[
            ("imputer", KNNImputer(n_neighbors=5)),
            ("feature_selection", SelectFromModel(RandomForestClassifier(random_state=SEED))),
            ("model", RandomForestClassifier(random_state=SEED))
        ])
    }

    # 특성 선정 후 어떤 것들이 선정됐는지 보고하기 위해 원본 저장
    features = X_train.columns.tolist()

    # 가장 유력한 모델을 선정하기 위한 변수 초기화
    best_model = ""
    best_recall = -1.0
    best_pipeline = None

    # 전역 설정을 무시하고 명시적으로 로컬 텍스트 방식 강제
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Heart Disease Classification")

    for model, pipeline in pipelines.items():
        with mlflow.start_run(run_name=model):
            # MLflow에 전처리 상태 아티팩트 기록
            mlflow.log_dict(train_bounds, "preprocessing_state/iqr_bounds.json")
            
            # 모델에 대한 5-fold 교차 검증
            cv_scores = cross_validate(
                pipeline, X_train_final, y_train,
                cv=5,
                scoring=['balanced_accuracy', 'precision', 'recall', 'f1']
            )
            
            # 교차 검증의 평균 지표 로깅
            mlflow.log_metrics({
                "cv_mean_balanced_accuracy": cv_scores['test_balanced_accuracy'].mean(),
                "cv_mean_precision": cv_scores['test_precision'].mean(),
                "cv_mean_recall": cv_scores['test_recall'].mean(),
                "cv_mean_f1": cv_scores['test_f1'].mean()
            })
            print(f"\n{model} CV 평균 Recall: {cv_scores['test_recall'].mean():.4f}\n")
            
            # CV 평균 Recall 저장
            mean_recall = cv_scores['test_recall'].mean()
            
            # 최고 성능 모델 갱신 (5-fold 교차 검증 평균 Recall 기반으로 결정)
            if mean_recall > best_recall:
                best_recall = mean_recall
                best_model = model
                best_pipeline = pipeline
            
            # 학습 데이터만 fit
            pipeline.fit(X_train_final, y_train)
            
            # 선정된 특성 추출 후 보고
            selector = pipeline.named_steps["feature_selection"]
            mask = selector.get_support()
            selected_features = [feature for feature, is_selected in zip(features, mask) if is_selected]
            
            print(f"[{model} 모델 학습 완료]")
            print(f"선정된 특성: {selected_features}\n")
            
            # 테스트 데이터로 예측
            y_pred = pipeline.predict(X_test_final)
            
            # 지표 계산 및 로깅
            # balanced accuracy, precision, recall, F1, confusion matrix 준비
            metrics = {
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0)
            }
            cm = confusion_matrix(y_test, y_pred)
            
            # 파라미터 로깅
            mlflow.log_params({
                "test_size": test_size,
                "random_state": SEED,
                "model": model,
            })
            
            # 평가 지표 로깅
            mlflow.log_metrics(metrics)
            
            # 모델 아티팩트 기록 및 모델 계열 태그 지정
            mlflow.sklearn.log_model(pipeline, "model")
            mlflow.set_tag("model_type", model)

            # 모든 모델에 대해 balanced accuracy, precision, recall, F1, confusion matrix 를 보고
            print("[MLflow 로깅 결과]")
            print(f"Balanced Accuracy: {metrics['balanced_accuracy']}")
            print(f"Precision: {metrics['precision']}")
            print(f"Recall: {metrics['recall']}")
            print(f"F1: {metrics['f1']}")
            print(f"Confusion Matrix:\n{cm}")

    # [최종 모델 선택 및 임상적 맥락에서 선택 사유 정당화]
    # 혼동 행렬에서 가장 중요한 값은 심장병 진단이라는 주제 특성을 반영하여 실제 환자를 정상으로 오진하는 False Negative이다.
    # 따라서 재현율(Recall)이 높은 모델이 가장 유력하다.
    # 5-fold 교차 검증을 거친 후 CV 평균 Recall이 가장 높은 것을 유력한 모델로 선정한다.
    # 현재 데이터셋으로는 SVC가 가장 유력한 모델이다.
    print(f"\n==============================\n선정된 가장 유력한 모델: {best_model} (재현율이 가장 높은 모델)\n==============================\n")

    # 세 모델 중 하나를 선정하므로, 각 모델에 맞는 탐색 공간 사전 정의
    param_grid = {
        "LogisticRegression": {
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'model__max_iter': [100, 200]
        },
        "SVC": {
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'model__kernel': ['linear', 'rbf']
        },
        "RandomForest": {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20]
        }
    }

    with mlflow.start_run(run_name=f"{best_model}_Tuning"):
        grid_search = GridSearchCV(
            estimator=best_pipeline,
            param_grid=param_grid[best_model],
            cv=5,
            scoring='recall',
        )
        
        # 튜닝 진행
        grid_search.fit(X_train_final, y_train)
        
        # 튜닝한 파이프라인
        tuned_pipeline = grid_search.best_estimator_
        
        # 파라미터 로깅
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("tuned_cv_best_recall", grid_search.best_score_)
        
        # 튜닝된 파이프라인을 사용하여 테스트 데이터 최종 예측
        y_pred_tuned = tuned_pipeline.predict(X_test_final)
        
        metrics_tuned = {
            "final_balanced_accuracy": balanced_accuracy_score(y_test, y_pred_tuned),
            "final_precision": precision_score(y_test, y_pred_tuned, zero_division=0),
            "final_recall": recall_score(y_test, y_pred_tuned, zero_division=0),
            "final_f1": f1_score(y_test, y_pred_tuned, zero_division=0)
        }
        cm_tuned = confusion_matrix(y_test, y_pred_tuned)
        
        mlflow.log_metrics(metrics_tuned)
        
        print(f"[{best_model} 튜닝 후 최종 결과]")
        print(f"최적 파라미터: {grid_search.best_params_}")
        print(f"최종 Recall: {metrics_tuned['final_recall']}")
        print(f"최종 혼동 행렬:\n{cm_tuned}")
        
        # 최종 평가 지표 계산 및 로깅
        mlflow.sklearn.log_model(tuned_pipeline, "best_model")
        mlflow.set_tag("model_type", f"{best_model}_Tuned")