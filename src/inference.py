import os
import sys
import json
import yaml
import pandas as pd
import mlflow
from preprocessing import impute_missing_values, IQRClamp

# MLflow 기록을 조회하여 가장 유력한 모델의 로컬 디렉토리와 전처리 기준 파일 경로를 반환
def get_best_model_artifacts():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment = mlflow.get_experiment_by_name("Heart Disease Classification")
    if not experiment:
        raise ValueError("MLflow 실험을 찾을 수 없습니다.")

    # 성능 지표(Recall) 기반으로 유력 모델 탐색
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.final_recall DESC"]
    )
    
    # final_recall 지표가 있는 Run만 필터링
    valid_runs = runs.dropna(subset=['metrics.final_recall'])
    if valid_runs.empty:
        raise ValueError("평가 지표가 기록된 유력 모델을 찾을 수 없습니다.")
        
    best_run_id = valid_runs.iloc[0].run_id
    print(f"선정된 Run ID: {best_run_id} (Recall: {valid_runs.iloc[0]['metrics.final_recall']})")
    
    # mlruns 디렉토리 탐색
    model_dir = None
    bounds_path = None
    
    for root, _, files in os.walk("mlruns"):
        # 모델 폴더 식별
        if "MLmodel" in files and not model_dir:
            try:
                with open(os.path.join(root, "MLmodel"), "r", encoding="utf-8") as f:
                    if yaml.safe_load(f).get("run_id") == best_run_id:
                        model_dir = root
            except Exception:
                pass
                
        # 전처리 기준 파일 식별
        if "iqr_bounds.json" in files and not bounds_path:
            bounds_path = os.path.join(root, "iqr_bounds.json")

    if not model_dir:
        raise FileNotFoundError(f"Run ID {best_run_id}에 해당하는 모델 폴더를 로컬에서 찾을 수 없습니다.")
    if not bounds_path:
        raise FileNotFoundError("전처리 기준 파일(iqr_bounds.json)을 찾을 수 없습니다.")
        
    return model_dir, bounds_path

def main():
    input_path = "data/batch_input.csv"
    output_path = "data/batch_predictions.csv"
    
    if not os.path.exists(input_path):
        print(f"입력 파일({input_path})이 존재하지 않습니다.")
        sys.exit(1)
        
    batch_df = pd.read_csv(input_path)
    print(f"배치 데이터 크기: {batch_df.shape}")

    try:
        # 아티팩트 경로 탐색
        model_dir, bounds_path = get_best_model_artifacts()
        
        # 기준값 로드 및 전처리 (결측치는 모델 내부의 파이프라인에서 자동으로 처리됨)
        with open(bounds_path, "r", encoding="utf-8") as f:
            train_bounds = json.load(f)
            
        batch_clamped = IQRClamp(batch_df, train_bounds)
        
        # 모델 로드 및 추론 수행
        model = mlflow.pyfunc.load_model(model_dir)
        batch_df['prediction'] = model.predict(batch_clamped)
        
        # 결과 저장
        batch_df.to_csv(output_path, index=False)
        print(f"=== 추론 완료: 결과가 {output_path}에 저장되었습니다. ===")
        
    except Exception as e:
        print(f"\n추론 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()