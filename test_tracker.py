import mmcv
import mmengine
import tempfile
from mmtrack.apis import inference_mot, init_model
from mmtrack.utils import register_all_modules
from mmtrack.registry import VISUALIZERS

register_all_modules(init_default_scope=True)

import numpy as np

CONFIG = "./configs/mot/qdtrack/qdtrack_yolox_s_mot17halftrain_test-mot17halfval.py"

model = init_model(CONFIG)

for i in range(10):
    result = inference_mot(model, np.random.rand(320, 320, 3), i)

print(result)
print("SUCC ASS")
