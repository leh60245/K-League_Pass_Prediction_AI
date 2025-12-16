import pandas as pd

v1 = pd.read_csv('submission_v1_final.csv')
v2 = pd.read_csv('submission_v2_20251216_162340.csv')

ens = pd.DataFrame({
    'game_episode': v1['game_episode'],
    'end_x': v1['end_x'] * 0.7 + v2['end_x'] * 0.3,
    'end_y': v1['end_y'] * 0.7 + v2['end_y'] * 0.3
})

ens.to_csv('submission_ensemble_v1_v2.csv', index=False)

print('✅ Ensemble 생성 완료!')
print(f'📊 총 예측: {len(ens):,}개')
print(f'📊 end_x 평균: {ens["end_x"].mean():.2f}m')
print(f'📊 end_y 평균: {ens["end_y"].mean():.2f}m')
print(f'📊 end_x 범위: [{ens["end_x"].min():.2f}, {ens["end_x"].max():.2f}]')
print(f'📊 end_y 범위: [{ens["end_y"].min():.2f}, {ens["end_y"].max():.2f}]')

