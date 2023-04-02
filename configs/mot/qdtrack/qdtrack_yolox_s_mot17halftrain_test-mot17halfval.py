_base_ = [
    './qdtrack_yolox_s_base.py',
    '../../_base_/datasets/mot_challenge.py',
]

# evaluator
val_evaluator = [
    dict(type='CocoVideoMetric', metric=['bbox'], classwise=True),
    dict(type='MOTChallengeMetrics', metric=['HOTA', 'CLEAR', 'Identity'])
]

test_evaluator = val_evaluator
