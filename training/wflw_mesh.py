# =====================================================================
#  wflw_mesh.py  -  우리가 학습한 WFLW 98점 모델로 얼굴 점 찾기 (Colab·보드 공용)
#
#  얼굴 검출: OpenCV YuNet (face_detection_yunet_2023mar.onnx, 같은 폴더)
#  점 찾기  : wflw_mesh.tflite (WFLW_Train.ipynb 가 만든 것, 같은 폴더)
#  반환     : (103,3) 픽셀 좌표  = WFLW 98점 + 가상 점 5개 (wflw_prep.add_virtual_points)
#
#  tflite 실행기는 tflite_runtime → ai_edge_litert → tensorflow 순서로 찾는다.
# =====================================================================
import os
import numpy as np
import cv2
from wflw_prep import crop_box, crop_face, to_input, pts_to_image, add_virtual_points

MESH_FILE = 'wflw_mesh.tflite'
DET_FILE = 'face_detection_yunet_2023mar.onnx'          # 대체: OpenCV YuNet (남이 학습한 검출기)
DET_URL = 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
OUR_DET_FILE = 'face_det_int8.tflite'                   # 우선: 우리가 WIDER FACE 로 학습한 YOLO11n (FaceDet_Train.ipynb). 있으면 이걸 씀
OUR_DET_SIZE, OUR_DET_CONF, OUR_DET_IOU = 320, 0.5, 0.45

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))        # 모델은 실행 위치가 아니라 이 파일이 있는 폴더에서 찾는다

def model_path(name):
    """파일 이름 → 절대 경로 (이미 절대 경로면 그대로). 현재 폴더와 스크립트 폴더 중 있는 쪽."""
    if os.path.isabs(name): return name
    for cand in (os.path.join(_MODEL_DIR, name), os.path.abspath(name)):
        if os.path.exists(cand): return cand
    return os.path.join(_MODEL_DIR, name)





def _interpreter(path):
    """tflite 실행기. 새 모델 형식을 읽을 수 있는 ai_edge_litert 를 먼저 찾고, 없으면 tflite_runtime, tensorflow 순."""
    try:
        from ai_edge_litert import interpreter as tfl       # pip install ai-edge-litert  (최신 모델 형식 지원)
    except ImportError:
        try:
            import tflite_runtime.interpreter as tfl        # 구형. Colab 최신 모델을 못 읽을 수 있음
        except ImportError:
            import tensorflow as tf
            tfl = tf.lite
    try:
        it = tfl.Interpreter(model_path=path, num_threads=4)
    except ValueError as error:
        if 'opcode' in str(error) or 'version' in str(error):
            raise RuntimeError(
                '%s 를 열 수 없습니다 (tflite 실행기가 오래됨: %s).\n'
                '  해결 1) 보드에서  pip install ai-edge-litert\n'
                '  해결 2) Colab WFLW_Train.ipynb 6번 셀에서 conv.optimizations 줄을 주석 처리하고 다시 변환'
                % (path, tfl.__name__)) from error
        raise
    it.allocate_tensors()
    return it


class YoloFaceDetector:
    """우리 YOLO 얼굴 검출기 (RPS rps_solo.py 의 detect() 와 같은 전처리·후처리). frame → [(x1,y1,x2,y2,score), …]"""

    def __init__(self, path=OUR_DET_FILE):
        path = model_path(path)
        self.it = _interpreter(path)
        self.inp = self.it.get_input_details()[0]; self.out = self.it.get_output_details()[0]
        self.in_scale, self.in_zero = self.inp['quantization']; self.out_scale, self.out_zero = self.out['quantization']
        shp = [int(v) for v in self.inp['shape']]                              # (1,320,320,3) NHWC 또는 (1,3,320,320) NCHW
        self.nchw = (shp[1] == 3 and shp[-1] != 3)
        self.size = shp[2] if self.nchw else shp[1]
        print('YoloFaceDetector: input', shp, 'NCHW' if self.nchw else 'NHWC', '| output', [int(v) for v in self.out['shape']])

    @staticmethod
    def letterbox(img, size):
        h, w = img.shape[:2]; r = min(size / w, size / h); nw, nh = int(w * r), int(h * r)
        px, py = (size - nw) // 2, (size - nh) // 2
        out = np.full((size, size, 3), 114, np.uint8); out[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh))
        return out, r, px, py

    @staticmethod
    def nms(boxes, scores, thr):
        if not boxes: return []
        b = np.array(boxes, np.float32); sc = np.array(scores); x1, y1, x2, y2 = b.T; area = (x2 - x1) * (y2 - y1)
        order = sc.argsort()[::-1]; keep = []
        while order.size:
            i = order[0]; keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            order = order[np.where(inter / (area[i] + area[order[1:]] - inter + 1e-6) <= thr)[0] + 1]
        return keep

    def __call__(self, bgr):
        H, W = bgr.shape[:2]
        lb, r, px, py = self.letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self.size)
        x = lb.astype(np.float32)[None] / 255.0
        if self.nchw: x = np.ascontiguousarray(x.transpose(0, 3, 1, 2))      # (1,3,H,W)
        if self.inp['dtype'] == np.int8: x = (x / self.in_scale + self.in_zero).astype(np.int8)
        self.it.set_tensor(self.inp['index'], x); self.it.invoke()
        raw = self.it.get_tensor(self.out['index'])[0]
        if self.out['dtype'] == np.int8: raw = (raw.astype(np.float32) - self.out_zero) * self.out_scale
        if raw.shape[0] <= 8: raw = raw.T                                     # (5,N) → (N,5) = cx cy w h score  (이미 (N,5) 면 그대로)
        boxes, scores = [], []
        for cx, cy, w, h, sc in raw[:, :5]:
            if sc < OUR_DET_CONF: continue
            cx, cy, w, h = cx * self.size, cy * self.size, w * self.size, h * self.size
            boxes.append([np.clip((cx - w / 2 - px) / r, 0, W), np.clip((cy - h / 2 - py) / r, 0, H),
                          np.clip((cx + w / 2 - px) / r, 0, W), np.clip((cy + h / 2 - py) / r, 0, H)])
            scores.append(float(sc))
        return [(*boxes[i], scores[i]) for i in self.nms(boxes, scores, OUR_DET_IOU)]


def _interpreter(path):
    """tflite 실행기. 새 모델 형식을 읽을 수 있는 ai_edge_litert 를 먼저 찾고, 없으면 tflite_runtime, tensorflow 순."""
    try:
        from ai_edge_litert import interpreter as tfl       # pip install ai-edge-litert  (최신 모델 형식 지원)
    except ImportError:
        try:
            import tflite_runtime.interpreter as tfl        # 구형. Colab 최신 모델을 못 읽을 수 있음
        except ImportError:
            import tensorflow as tf
            tfl = tf.lite
    try:
        it = tfl.Interpreter(model_path=path, num_threads=4)
    except ValueError as error:
        if 'opcode' in str(error) or 'version' in str(error):
            raise RuntimeError(
                '%s 를 열 수 없습니다 (tflite 실행기가 오래됨: %s).\n'
                '  해결 1) 보드에서  pip install ai-edge-litert\n'
                '  해결 2) Colab WFLW_Train.ipynb 6번 셀에서 conv.optimizations 줄을 주석 처리하고 다시 변환'
                % (path, tfl.__name__)) from error
        raise
    it.allocate_tensors()
    return it




class WflwMesh:
    def __init__(self, mesh=MESH_FILE, detector=DET_FILE, score_thr=0.7):
        mesh, detector, our_det = model_path(mesh), model_path(detector), model_path(OUR_DET_FILE)
        if not os.path.exists(mesh):
            raise FileNotFoundError('%s 없음. colab 결과(WFLW_Train.ipynb)를 같은 폴더에 두세요' % mesh)
        self.it = _interpreter(mesh)
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]
        self.last_box = None
        # 얼굴 검출기: 우리 YOLO(face_det_int8.tflite) 가 있으면 그것, 없으면 OpenCV YuNet
        self.yolo = None; self.det = None
        if os.path.exists(our_det):
            self.yolo = YoloFaceDetector(our_det); det_name = 'our YOLO (%s)' % our_det
        elif os.path.exists(detector):
            self.det = cv2.FaceDetectorYN.create(detector, '', (320, 320), score_threshold=score_thr)
            self.det_size = (320, 320); det_name = 'YuNet (%s)' % detector
        else:
            raise FileNotFoundError('얼굴 검출기 없음: %s (FaceDet_Train.ipynb) 또는 %s (%s)' % (our_det, detector, DET_URL))
        print('WflwMesh 로딩 완료:', mesh, '| 얼굴 검출:', det_name)

    def detect(self, bgr):
        """가장 큰 얼굴 박스 (x1,y1,x2,y2) 또는 None"""
        if self.yolo is not None:
            dets = self.yolo(bgr)
            if not dets: return None
            x1, y1, x2, y2, _ = max(dets, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
            return float(x1), float(y1), float(x2), float(y2)
        h, w = bgr.shape[:2]
        if (w, h) != self.det_size:
            self.det.setInputSize((w, h)); self.det_size = (w, h)
        _, faces = self.det.detect(bgr)
        if faces is None or len(faces) == 0: return None
        x, y, bw, bh = max(faces, key=lambda f: f[2] * f[3])[:4]
        return float(x), float(y), float(x + bw), float(y + bh)

    def landmarks(self, bgr, box):
        """박스 하나 → (98,2) 픽셀 좌표"""
        x0, y0, s = crop_box(*box)
        face = crop_face(bgr, x0, y0, s)
        self.it.set_tensor(self.inp['index'], to_input(face)); self.it.invoke()
        p01 = self.it.get_tensor(self.out['index'])[0].reshape(98, 2)
        return pts_to_image(p01, x0, y0, s)

    def __call__(self, bgr):
        """프레임 → (103,3) 픽셀 좌표 또는 None"""
        box = self.detect(bgr)
        if box is None:
            self.last_box = None; return None
        self.last_box = box
        return add_virtual_points(self.landmarks(bgr, box))
