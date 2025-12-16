"""
결과 저장 유틸리티

모델별 결과를 시간별 폴더에 체계적으로 저장
"""

import os
from datetime import datetime
import shutil
import json


class ResultManager:
    """결과 관리 클래스"""
    
    def __init__(self, base_dir='results'):
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def create_model_dir(self, model_name):
        """모델별 타임스탬프 폴더 생성"""
        model_dir = os.path.join(self.base_dir, model_name, self.timestamp)
        os.makedirs(model_dir, exist_ok=True)
        return model_dir
    
    def save_submission(self, submission_df, model_name, metadata=None):
        """제출 파일 저장"""
        model_dir = self.create_model_dir(model_name)
        
        # CSV 저장
        csv_path = os.path.join(model_dir, 'submission.csv')
        submission_df.to_csv(csv_path, index=False)
        
        # 메타데이터 저장
        if metadata:
            meta_path = os.path.join(model_dir, 'metadata.json')
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        # 요약 정보 저장
        summary_path = os.path.join(model_dir, 'summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"모델: {model_name}\n")
            f.write(f"생성 시간: {self.timestamp}\n")
            f.write(f"예측 개수: {len(submission_df)}\n")
            f.write(f"평균 end_x: {submission_df['end_x'].mean():.2f}\n")
            f.write(f"평균 end_y: {submission_df['end_y'].mean():.2f}\n")
            f.write(f"end_x 범위: [{submission_df['end_x'].min():.2f}, {submission_df['end_x'].max():.2f}]\n")
            f.write(f"end_y 범위: [{submission_df['end_y'].min():.2f}, {submission_df['end_y'].max():.2f}]\n")
            if metadata:
                f.write(f"\n메타데이터:\n")
                for key, value in metadata.items():
                    f.write(f"  {key}: {value}\n")
        
        print(f"\n✅ 결과 저장 완료:")
        print(f"   📁 폴더: {model_dir}")
        print(f"   📄 파일:")
        print(f"      - submission.csv")
        print(f"      - summary.txt")
        if metadata:
            print(f"      - metadata.json")
        
        return model_dir
    
    def copy_model(self, model_path, model_name):
        """모델 파일도 함께 복사"""
        if not os.path.exists(model_path):
            return
        
        model_dir = os.path.join(self.base_dir, model_name, self.timestamp)
        os.makedirs(model_dir, exist_ok=True)
        
        dest_path = os.path.join(model_dir, os.path.basename(model_path))
        shutil.copy2(model_path, dest_path)
        print(f"   📦 모델 복사: {os.path.basename(model_path)}")
    
    def list_results(self, model_name=None):
        """저장된 결과 목록 출력"""
        if model_name:
            model_dirs = [os.path.join(self.base_dir, model_name)]
        else:
            model_dirs = [os.path.join(self.base_dir, d) 
                         for d in os.listdir(self.base_dir) 
                         if os.path.isdir(os.path.join(self.base_dir, d))]
        
        print("\n" + "=" * 80)
        print("  저장된 결과 목록")
        print("=" * 80)
        
        for model_dir in sorted(model_dirs):
            if not os.path.exists(model_dir):
                continue
                
            model_name = os.path.basename(model_dir)
            print(f"\n📊 {model_name}:")
            
            timestamps = [d for d in os.listdir(model_dir) 
                         if os.path.isdir(os.path.join(model_dir, d))]
            
            for ts in sorted(timestamps, reverse=True):
                ts_dir = os.path.join(model_dir, ts)
                summary_file = os.path.join(ts_dir, 'summary.txt')
                
                if os.path.exists(summary_file):
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()
                    print(f"   {ts} - {first_line}")
                else:
                    print(f"   {ts}")


def save_model_results(submission_df, model_name, val_score=None, 
                       train_score=None, weights=None, n_estimators=None):
    """
    편리한 결과 저장 함수
    
    Args:
        submission_df: 제출 DataFrame
        model_name: 모델 이름 (xgboost, lightgbm, ensemble 등)
        val_score: 검증 점수
        train_score: 학습 점수
        weights: 앙상블 가중치
        n_estimators: estimator 개수
    """
    manager = ResultManager()
    
    # 메타데이터 구성
    metadata = {
        'model_name': model_name,
        'timestamp': manager.timestamp,
        'n_predictions': len(submission_df)
    }
    
    if val_score is not None:
        metadata['val_score'] = val_score
    if train_score is not None:
        metadata['train_score'] = train_score
    if weights is not None:
        metadata['ensemble_weights'] = weights
    if n_estimators is not None:
        metadata['n_estimators'] = n_estimators
    
    # 저장
    model_dir = manager.save_submission(submission_df, model_name, metadata)
    
    # 모델 파일도 복사
    model_file = f"{model_name}.pkl"
    if os.path.exists(model_file):
        manager.copy_model(model_file, model_name)
    
    return model_dir


if __name__ == "__main__":
    # 테스트
    manager = ResultManager()
    manager.list_results()

