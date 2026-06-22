# 전처리 모듈
# 필요한 곳에서 import하여 사용합니다.

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# 이상치 처리 함수 (IQR 클램핑): 이상치는 최소/최대치로 제한됨
def IQRClamp(df, bounds=None):
    df_clamped = df.copy()
    result_bounds = {}
    
    # 숫자형 열만 사용
    numeric_cols = df_clamped.select_dtypes(include=[np.number]).columns
    
    # 각 열의 IQR 범위 계산
    for col in numeric_cols:
        # 입력된 bounds가 없을 때는 직접 계산 (None)
        if bounds is None:
            Q1 = df_clamped[col].quantile(0.25)
            Q3 = df_clamped[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            result_bounds[col] = (lower, upper)
        # 입력된 bounds가 있을 때는 그걸 사용
        else:
            lower, upper = bounds.get(col, (-np.inf, np.inf)) # bounds 없는 열은 제한 없음
        
        # 각 열을 각자의 IQR 범위로 클램핑
        df_clamped[col] = df_clamped[col].clip(lower=lower, upper=upper)
    
    if bounds is None:
        return df_clamped, result_bounds
    else:
        return df_clamped

# 빈 열 및 중복 행 제거
def clean_data(df):
    df_cleaned = df.dropna(axis=1, how='all')
    df_cleaned = df_cleaned.drop_duplicates()
    return df_cleaned