# =====================================================================
#  eye_prep.py  -  눈 정밀 모델(2단계)의 크롭 규칙.  Colab(학습)과 보드(실시간) 공용
#
#  1단계 wflw_mesh 가 찍은 98점은 눈·눈썹이 '평균 모양'으로 뭉개지는 경향이 있다 (GAP+Dense 회귀 구조의 한계).
#  그래서 1단계 결과로 눈+눈썹 주변만 잘라 작은 2단계 모델(eye_refine.tflite)에 다시 넣어
#  눈 8점 + 눈썹 9점 + 눈동자 1점 = 18점을 정밀하게 다시 찍는다.
#
#  !! 이 파일은 colab/ 과 보드(face_fortune.py 안에 복사된 구역)에 똑같아야 한다.
#  오른쪽 눈은 크롭을 좌우 반전해서 '왼쪽 눈'으로 만들어 넣고, 결과를 다시 반전한다 (모델은 왼쪽 눈만 배운다).
# =====================================================================
import numpy as np
import cv2

EYE_IMG = 64            # 2단계 모델 입력 한 변
EYE_MARGIN = 1.7        # 눈+눈썹 18점이 차지하는 긴 변 × 1.7 을 정사각 한 변으로

# 왼쪽(사진 기준 왼쪽) 눈썹 33~41, 눈 60~67, 눈동자 96  →  모델이 예측하는 18점 (이 순서)
LEFT_IDS = list(range(33, 42)) + list(range(60, 68)) + [96]
# 오른쪽은 wflw_prep.FLIP_PAIRS 와 같은 짝 (33↔46 … 41↔47, 60↔72 … 67↔73, 96↔97)
RIGHT_IDS = [46, 45, 44, 43, 42, 50, 49, 48, 47] + [72, 71, 70, 69, 68, 75, 74, 73] + [97]
N_EYE_PTS = len(LEFT_IDS)       # 18

EYE_IDS = {"left": LEFT_IDS, "right": RIGHT_IDS}


def eye_crop_box(pts, side, margin=EYE_MARGIN):
    """98점(또는 103점) 픽셀 좌표 + 'left'/'right' → 정사각 크롭 (x0, y0, 한 변 s)"""
    p = np.asarray(pts, np.float32)[EYE_IDS[side], :2]
    x1, y1 = p.min(0); x2, y2 = p.max(0)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    s = max(x2 - x1, y2 - y1, 8.0) * margin
    return float(cx - s / 2.0), float(cy - s / 2.0), float(s)


def crop_eye(img, x0, y0, s, side):
    """원본에서 (x0,y0,s) 정사각을 잘라 EYE_IMG×EYE_IMG 로. 오른쪽 눈이면 좌우 반전해서 '왼쪽 눈' 모양으로."""
    pad = int(s) + 10
    p = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    x0i, y0i, si = int(round(x0)) + pad, int(round(y0)) + pad, max(int(round(s)), 1)
    crop = cv2.resize(p[y0i:y0i + si, x0i:x0i + si], (EYE_IMG, EYE_IMG), interpolation=cv2.INTER_AREA)
    return crop[:, ::-1].copy() if side == "right" else crop


def eye_to_input(crop_bgr):
    """(EYE_IMG,EYE_IMG,3) uint8 BGR → (1,EYE_IMG,EYE_IMG,3) float32 0~1 RGB"""
    return (crop_bgr[:, :, ::-1].astype(np.float32) / 255.0)[None]


def eye_pts_to_crop(pts, x0, y0, s, side):
    """(학습용) 원본 픽셀 좌표의 18점 → 크롭 기준 0~1 좌표. 오른쪽이면 x 를 반전."""
    p = (np.asarray(pts, np.float32)[EYE_IDS[side], :2] - [x0, y0]) / s
    if side == "right":
        p[:, 0] = 1.0 - p[:, 0]
    return p


def eye_pts_to_image(p01, x0, y0, s, side):
    """모델 출력 (18,2) 0~1 → 원본 픽셀 좌표 (18,2). 오른쪽이면 x 를 되돌린다."""
    p = np.asarray(p01, np.float32).reshape(N_EYE_PTS, 2).copy()
    if side == "right":
        p[:, 0] = 1.0 - p[:, 0]
    return p * s + np.array([x0, y0], np.float32)
