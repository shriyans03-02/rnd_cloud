import sys
import numpy as np

print('Python:', sys.version)
try:
    import torch
    print('torch:', torch.__version__)
    print('torch.cuda.is_available:', torch.cuda.is_available())
    print('torch.cuda.device_count:', torch.cuda.device_count())
except Exception as e:
    print('torch import failed:', repr(e))

try:
    import onnxruntime as ort
    print('onnxruntime:', ort.__version__)
    print('ORT providers:', ort.get_available_providers())
except Exception as e:
    print('onnxruntime import failed:', repr(e))

try:
    from insightface.app import FaceAnalysis
except Exception as e:
    print('insightface import failed:', repr(e))
    raise SystemExit(1)

provider_sets = []
try:
    import torch
    import onnxruntime as ort
    if torch.cuda.is_available() and 'CUDAExecutionProvider' in ort.get_available_providers():
        provider_sets.append((['CUDAExecutionProvider', 'CPUExecutionProvider'], 0, 'cuda'))
except Exception:
    pass
provider_sets.append((['CPUExecutionProvider'], -1, 'cpu'))

last = None
for providers, ctx_id, label in provider_sets:
    try:
        print('\nTrying InsightFace provider:', label, providers)
        try:
            app = FaceAnalysis(name='buffalo_l', providers=providers, allowed_modules=['detection', 'recognition'])
        except TypeError:
            app = FaceAnalysis(name='buffalo_l', providers=providers)
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = app.get(img)
        print('InsightFace OK provider:', label, 'faces_on_blank:', len(faces))
        raise SystemExit(0)
    except Exception as e:
        last = e
        print('FAILED provider:', label, repr(e))

print('\nInsightFace failed for all providers. Last error:', repr(last))
raise SystemExit(2)
