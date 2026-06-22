import sys
import os

# 현재 파일 기준으로 부모 경로의 src 폴더를 탐색 경로에 추가
# 다음 줄은 AI로 생성된 코드입니다.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from preprocessing import clean_data, IQRClamp
from train import SEED

from sklearn.impute import KNNImputer

class TestModelInference(unittest.TestCase):
    # 초기화
    def setUp(self):
        # 테스트에서 쓸 훈련용 더미 데이터 생성
        self.X_train = pd.DataFrame({
            'age': [45, 50, 65, 30, 55],
            'cp': [0, 1, 2, 0, 3],
            'chol': [200, 250, 220, 180, 280]
        })
        self.y_train = np.array([0, 1, 1, 0, 1])
        
        # 전처리 함수 적용
        self.X_train_clamped, self.train_bounds = IQRClamp(self.X_train)
        
        # 테스트용 파이프라인 생성
        self.pipeline = Pipeline(steps=[
            ("imputer", KNNImputer(n_neighbors=2)),
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(random_state=SEED))
        ])
        
        self.pipeline.fit(self.X_train_clamped, self.y_train)
        
    # 예측 결과의 shape이 입력 shape와 일치하는지 테스트
    def test_prediction_output_shape(self):
        X_test = pd.DataFrame({
            'age': [60, 40], 'cp': [1, 0], 'chol': [240, 190]
        })
        
        X_test_clamped = IQRClamp(X_test, self.train_bounds)
        predictions = self.pipeline.predict(X_test_clamped)
        
        self.assertEqual(len(X_test), len(predictions))
        
    # 예측 확률이 [0, 1] 범위 내에 있고 행마다 합이 약 1인지 테스트
    def test_probability_range(self):
        X_test = pd.DataFrame({'age': [50, 45], 'cp': [2, 1], 'chol': [210, 250]})
        X_test_clamped = IQRClamp(X_test, self.train_bounds)
        
        probabilities = self.pipeline.predict_proba(X_test_clamped)
        
        # 모든 확률이 [0, 1] 범위 내에 있는지 확인
        self.assertTrue(np.all(probabilities >= 0.0) and np.all(probabilities <= 1.0))
        
        # 각 행의 확률 합이 1.0에 가까운지 확인
        row_sums = np.sum(probabilities, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)
        
    # 임상적으로 범위가 정해진 특성에 대한 입력값 범위 검증
    def test_input_values(self):
        X_test = pd.DataFrame({
            'age': [150, -4], 'cp': [0, 0], 'chol': [99999.9, -10.0]
        })
        
        try:
            X_clamped = IQRClamp(X_test, self.train_bounds)
            prediction = self.pipeline.predict(X_clamped)
            self.assertTrue(prediction[0] in [0, 1])
        except Exception as e:
            self.fail(f"임상 범위 초과 데이터 처리 테스트 중 에러 발생: {e}\n")
        
    # 고정 시드에서 파이프라인이 결정론적인지 테스트
    def test_deterministic_behavior(self):
        X_test = pd.DataFrame({
            'age': [45, 55], 'cp': [1, 2], 'chol': [220, 260]
        })
        
        X_test_clamped = IQRClamp(X_test, self.train_bounds)
        
        pred1 = self.pipeline.predict(X_test_clamped)
        pred2 = self.pipeline.predict(X_test_clamped)
        
        np.testing.assert_array_equal(pred1, pred2)
        
if __name__ == '__main__':
    unittest.main()