# =====================================================================
#  wflw_prep.py  -  얼굴 박스 → 정사각 크롭 → 모델 입력.  Colab(학습)과 보드(실시간) 공용
#
#  !! 이 파일은 colab/ 과 board/ 에 똑같이 있어야 한다 (학습 때 자른 방식 == 실행 때 자른 방식)
#  크롭 규칙이 1픽셀이라도 다르면 점이 통째로 밀린다.
# =====================================================================
import numpy as np
import cv2

IMG = 160          # 모델 입력 한 변 (학습·보드 동일)
MARGIN = 1.2       # 얼굴 박스 긴 변 × 1.2 를 정사각 한 변으로

# WFLW 98점 좌우 반전 쌍 (학습 증강용). 사진 기준 왼쪽 ↔ 오른쪽
FLIP_PAIRS = [(0,32),(1,31),(2,30),(3,29),(4,28),(5,27),(6,26),(7,25),(8,24),(9,23),(10,22),(11,21),(12,20),(13,19),(14,18),(15,17),
              (33,46),(34,45),(35,44),(36,43),(37,42),(38,50),(39,49),(40,48),(41,47),
              (55,59),(56,58),
              (60,72),(61,71),(62,70),(63,69),(64,68),(65,75),(66,74),(67,73),
              (76,82),(77,81),(78,80),(87,83),(86,84),(88,92),(89,91),(95,93),
              (96,97)]
FLIP_PERM = np.arange(98)
for _a, _b in FLIP_PAIRS:
    FLIP_PERM[_a], FLIP_PERM[_b] = _b, _a


def crop_box(x1, y1, x2, y2, margin=MARGIN):
    """얼굴 박스(x1,y1,x2,y2) → 정사각 크롭 (x0, y0, 한 변 s). 좌표계는 원본 픽셀."""
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    s = max(x2 - x1, y2 - y1) * margin
    return cx - s / 2.0, cy - s / 2.0, s


def crop_face(img, x0, y0, s):
    """원본 이미지에서 (x0,y0,s) 정사각을 잘라 IMG×IMG 로. 화면 밖은 검정으로 채움."""
    pad = int(s) + 10
    p = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    x0i, y0i, si = int(round(x0)) + pad, int(round(y0)) + pad, max(int(round(s)), 1)
    return cv2.resize(p[y0i:y0i + si, x0i:x0i + si], (IMG, IMG), interpolation=cv2.INTER_AREA)


def to_input(face_bgr):
    """크롭된 (IMG,IMG,3) uint8 BGR → 모델 입력 (1,IMG,IMG,3) float32 0~1 (RGB)"""
    return (face_bgr[:, :, ::-1].astype(np.float32) / 255.0)[None]


def pts_to_image(pts01, x0, y0, s):
    """모델 출력 (98,2) 0~1 → 원본 픽셀 좌표 (98,2)"""
    return np.asarray(pts01, np.float32) * s + np.array([x0, y0], np.float32)


# ---------------------------------------------------------------- 가상 점 (WFLW 에 없는 점을 기하학으로 만든다)
# face_metrics.P 는 98점 + 아래 5개 가상 점(98~102)을 쓴다.
V_FOREHEAD_TOP, V_TEMPLE_L, V_TEMPLE_R, V_UNDER_EYE_L, V_UNDER_EYE_R = 98, 99, 100, 101, 102
N_POINTS = 103

def add_virtual_points(p98):
    """(98,2|3) 픽셀 좌표 → (103,3). z 는 전부 0 (2D 모델)
       98 forehead_top : 미간(51)에서 코 길이(51→57)만큼 위  (헤어라인 분할이 있으면 그 값으로 덮어씀)
       99,100 temple   : 눈썹 바깥 끝(33,46)에서 코 길이 × 0.6 만큼 위
       101,102 under_eye: 아래 눈꺼풀(66,74)에서 눈 높이만큼 아래"""
    p = np.asarray(p98, np.float32)
    xy = p[:, :2]
    nose_vec = xy[51] - xy[57]                     # 코밑 → 미간 (위쪽 방향)
    forehead = xy[51] + nose_vec
    temple_l = xy[33] + nose_vec * 0.6
    temple_r = xy[46] + nose_vec * 0.6
    under_l = xy[66] + (xy[66] - xy[62])
    under_r = xy[74] + (xy[74] - xy[70])
    extra = np.stack([forehead, temple_l, temple_r, under_l, under_r])
    out = np.zeros((N_POINTS, 3), np.float32)
    out[:98, :2] = xy
    if p.shape[1] >= 3: out[:98, 2] = p[:, 2]
    out[98:, :2] = extra
    return out
