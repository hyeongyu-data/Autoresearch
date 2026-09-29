"""후보 예측을 고정된 평가 계약으로 채점하는 연구 Harness 구성요소.

[파이프라인] 후보 workspace가 예측 산출물을 제출한 뒤, 후보 코드와 분리된 Judge가
정답·평가 정책을 읽어 품질 판정으로 바꾸는 구간을 담당한다.

[기능] 예측 CSV의 행 식별자·수치·중복·누락을 검증하고 ROC-AUC/RMSE를 계산한다.
학습, 후보 코드 실행, Kubernetes 오케스트레이션, 결과 저장소와 운영 승격은 이
패키지의 책임이 아니다.
"""

from autoresearch.research_harness.judge import (
    EvaluationManifest,
    JudgeError,
    JudgeResult,
    JudgePolicy,
    judge_predictions,
)

__all__ = [
    "EvaluationManifest",
    "JudgeError",
    "JudgePolicy",
    "JudgeResult",
    "judge_predictions",
]
