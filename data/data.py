# 데이터셋을 읽어와 타깃 이진화 및 csv 저장을 수행하는 코드

from ucimlrepo import fetch_ucirepo
import os
import pandas as pd

# 데이터셋 준비
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets

# X와 y를 하나의 데이터프레임으로 병합
df = pd.concat([X, y], axis=1)

# 심장병 발병 여부를 나타내는 열 이름이 num으로 되어 있으므로 target(타깃)으로 변경
df.rename(columns={'num': 'target'}, inplace=True)

# 타깃 값이 0~4로 다중 클래스이므로 0은 0(정상)으로, 1~4는 모두 1(심장병)로 통합하여 이진화
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# 데이터를 스크립트 위치에 .csv 파일로 저장
dir_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(dir_path, 'heart_disease.csv')

df.to_csv(file_path, index=False)

print(f"{file_path} 경로로 데이터를 저장했습니다.")