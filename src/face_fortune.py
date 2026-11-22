"""Face Fortune - single file (FaceReading_3 / WFLW, 측정값만).

FaceReading_3/board 의 다섯 모듈을 한 파일로 합친 것:
    wflw_prep    얼굴 크롭 규칙 · 좌우 반전 쌍 · 가상 점 5개            (Colab 학습과 같은 규칙이어야 함)
    face_metrics WFLW 98점(+가상 5점) → 관상용 비율 42개
    hairline     우리 분할 모델(hair_seg.tflite)로 실제 헤어라인 (선택, LiteRT 로 실행)
    wflw_mesh    얼굴 검출(우리 YOLO 또는 YuNet) + wflw_mesh.tflite 로 98점 찾기
    face_fortune 촬영 루프: 30프레임 수집, 정면·시선·깜빡임 검사, 안정되면 자동 저장

같은 폴더에 필요한 모델 파일 (코드로 합칠 수 없는 것):
    wflw_mesh.tflite                    우리가 WFLW 로 학습한 98점 모델
    face_det_int8.tflite                우리가 WIDER FACE 로 학습한 YOLO 얼굴 검출기 (FaceDet_Train.ipynb). 없으면 ↓
    face_detection_yunet_2023mar.onnx   OpenCV YuNet 얼굴 검출기 (대체)
    hair_seg.tflite                     (선택) 우리가 CelebAMask-HQ 로 학습한 헤어라인 분할 모델. 없으면 가상 이마 점 사용
    eye_refine.tflite                   (선택) 2단계 눈/눈썹 정밀 보정 모델. 없으면 1단계 WFLW 눈 점 그대로 사용
    face_reference_10level_clean.csv    (선택) 42개 측정값의 사람 기준 p10~p90. 없거나 잘못돼도 촬영은 계속됨

실행:  python face_fortune.py            (카메라 기본 1번, 안 열리면 0·2번 자동 시도)
키  :  R 새로 스캔 / S 다시 저장 / Q 종료 / 1 관상 / 2 닮은꼴
출력:  captures/YYYYMMDD_HHMMSS/  face_measurement.jpg, measurement_screen.jpg,
                                  vision_landmarks.json, analysis_result.json, handoff.json(다음 단계로 넘길 것),
                                  face_reference_10level.json, cnn_frame_01.jpg ... cnn_frame_05.jpg,
                                  cnn_face_01.jpg ... cnn_face_05.jpg,
                                  cnn_face_v41_01.jpg ... cnn_face_v41_05.jpg

!! wflw_prep 부분(크롭 규칙·IMG·MARGIN)을 고치면 Colab 에서 모델을 다시 학습해야 한다.
!! face_metrics 부분을 고치면 (관상 카드를 쓸 경우) norms.json 을 다시 만들어야 한다.
"""

from __future__ import annotations

import math
import os
import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import cv2

from run_team_animal_test import (
    analyze_capture_folder,
)


# =============================================================================
#  [1] wflw_prep  - 크롭 규칙 · 반전 쌍 · 가상 점
# =============================================================================

# =====================================================================
#  wflw_prep.py  -  얼굴 박스 → 정사각 크롭 → 모델 입력.  Colab(학습)과 보드(실시간) 공용
#
#  !! 이 파일은 colab/ 과 board/ 에 똑같이 있어야 한다 (학습 때 자른 방식 == 실행 때 자른 방식)
#  크롭 규칙이 1픽셀이라도 다르면 점이 통째로 밀린다.
# =====================================================================

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


# =============================================================================
#  [2] face_metrics - 98점 → 비율 42개
# =============================================================================

# =====================================================================
#  face_metrics.py  -  WFLW 98점(+가상 점 5개) → 관상용 측정값   [FaceReading_3 / 방법 A]
#  Colab(기준표 생성)과 보드(실시간)에서 같은 파일을 그대로 씀  !! colab/ 과 board/ 가 동일해야 함
#
#  - 입력 pts: wflw_mesh.WflwMesh 가 돌려주는 (103,3) 픽셀 좌표 (z 는 0)
#  - 모든 길이는 두 눈동자(96,97) 사이 거리(IPD)로 나눈 비율, 각도는 도(°)
#  - MediaPipe 판과 다른 점: z 가 없어 cheek_prom / nose_tip_h / forehead_tilt 삭제,
#    콧대 양옆 점이 없어 bridge_w 삭제 → 측정값 46 → 42개
# =====================================================================

# ---- WFLW 98점 번호 (+ wflw_prep.add_virtual_points 가 붙이는 가상 점 98~102) ----
#  "왼쪽" 은 사진 기준 왼쪽
P = dict(
    forehead_top=98, chin=16, glabella=51, subnasale=57,     # 98 = 가상(헤어라인 분할이 있으면 그 값으로 교체)
    cheek_l=0, cheek_r=32,              # 얼굴 폭 (광대 바깥, 귀 높이)
    temple_l=99, temple_r=100,          # 이마 폭 (가상)
    jaw_l=5, jaw_r=27,                  # 턱뼈 (하악각)
    chin_l=13, chin_r=19,               # 턱끝 폭
    eye_l_out=60, eye_l_in=64, eye_r_in=68, eye_r_out=72,
    eye_l_top=62, eye_l_bot=66, eye_r_top=70, eye_r_bot=74,
    iris_l=96, iris_r=97,               # 눈동자
    brow_l_out=33, brow_l_in=37, brow_r_in=42, brow_r_out=46,
    nose_bridge=51, nose_tip=54, ala_l=55, ala_r=59,
    mouth_l=76, mouth_r=82,
    lip_top=79, lip_up_bot=89, lip_low_top=93, lip_bot=85,
    under_eye_l=101, under_eye_r=102,   # 가상
)

# 좌우 대칭 계산용 쌍 (왼쪽, 오른쪽)
SYM_PAIRS = [(60,72),(64,68),(33,46),(37,42),(55,59),(76,82),(0,32),(5,27),(13,19),(96,97)]

# 측정값 이름 (순서 고정: CSV 열, MLP 입력 순서로 그대로 사용)
METRIC_NAMES = [
    'face_ratio','forehead_w','cheek_w','jaw_w','chin_w','chin_len','jaw_angle',
    'eye_len','eye_aspect','eye_gap','eye_tilt','under_eye',
    'brow_len','brow_eye_gap','brow_tilt','glabella_w',
    'nose_len','ala_w','nose_taper',
    'mouth_w','lip_up','lip_low','mouth_tilt','philtrum',
    'symmetry',
    'zone_top','zone_mid','zone_bot',
    # 팀 측정표 추가분 (좌우 따로 · 윤곽 비율 · 입)
    'eye_len_l','eye_len_r','eye_h_l','eye_h_r','brow_len_l','brow_len_r','jaw_angle_l','jaw_angle_r',
    'upper_cheek_ratio','jaw_cheek_ratio','chin_jaw_ratio','jaw_taper',
    'lip_h','mouth_hw',
]

# 한글 이름 (결과 카드용)
METRIC_KO = {
    'face_ratio':'얼굴 길쭉한 정도','forehead_w':'이마 폭','cheek_w':'광대 폭','jaw_w':'턱뼈 폭','chin_w':'턱끝 폭',
    'chin_len':'턱 길이','jaw_angle':'턱선 각도',
    'eye_len':'눈 길이','eye_aspect':'눈 가로세로 비율','eye_gap':'두 눈 사이 간격','eye_tilt':'눈꼬리 기울기','under_eye':'눈 밑 도톰함',
    'brow_len':'눈썹 길이','brow_eye_gap':'눈썹~눈 사이','brow_tilt':'눈썹 기울기','glabella_w':'미간 폭',
    'nose_len':'코 길이','ala_w':'콧볼 폭','nose_taper':'코 갸름한 정도',
    'mouth_w':'입 크기','lip_up':'윗입술 두께','lip_low':'아랫입술 두께','mouth_tilt':'입꼬리 기울기','philtrum':'인중 길이',
    'symmetry':'좌우 대칭',
    'zone_top':'위(이마) 구역 %','zone_mid':'가운데(눈~코) 구역 %','zone_bot':'아래(코~턱) 구역 %',
    'eye_len_l':'왼쪽 눈 길이','eye_len_r':'오른쪽 눈 길이','eye_h_l':'왼쪽 눈 높이','eye_h_r':'오른쪽 눈 높이',
    'brow_len_l':'왼쪽 눈썹 길이','brow_len_r':'오른쪽 눈썹 길이','jaw_angle_l':'왼쪽 턱선 각도','jaw_angle_r':'오른쪽 턱선 각도',
    'upper_cheek_ratio':'이마 폭/광대 폭','jaw_cheek_ratio':'턱뼈 폭/광대 폭','chin_jaw_ratio':'턱끝 폭/턱뼈 폭','jaw_taper':'턱 좁아짐',
    'lip_h':'입술 전체 두께','mouth_hw':'입 세로/가로 비율',
}

# 부위 묶음 (결과 카드 표 구성용)
METRIC_GROUPS = [
    ('얼굴 윤곽', ['face_ratio','forehead_w','cheek_w','jaw_w','chin_w','chin_len','jaw_angle']),
    ('눈',       ['eye_len','eye_aspect','eye_gap','eye_tilt','under_eye']),
    ('눈썹',     ['brow_len','brow_eye_gap','brow_tilt','glabella_w']),
    ('코',       ['nose_len','ala_w','nose_taper']),
    ('입',       ['mouth_w','lip_up','lip_low','mouth_tilt','philtrum']),
    ('그 밖에',  ['symmetry']),
    ('윤곽 비율', ['upper_cheek_ratio','jaw_cheek_ratio','chin_jaw_ratio','jaw_taper']),
    ('좌우 따로', ['eye_len_l','eye_len_r','eye_h_l','eye_h_r','brow_len_l','brow_len_r','jaw_angle_l','jaw_angle_r']),
    ('입 (추가)', ['lip_h','mouth_hw']),
]

# 5단계 라벨 (백분위 기준). 값이 클수록 오른쪽 라벨
LEVELS = ['아주 %s 편', '%s 편', '보통', '%s 편', '아주 %s 편']
LEVEL_WORDS = {   # (작을 때 단어, 클 때 단어)
    'face_ratio':('둥근','긴'),'forehead_w':('좁은','넓은'),'cheek_w':('좁은','넓은'),'jaw_w':('좁은','넓은'),'chin_w':('뾰족한','넓은'),
    'chin_len':('짧은','긴'),'jaw_angle':('각진','둥근'),
    'eye_len':('짧은','긴'),'eye_aspect':('동그란','가늘고 긴'),'eye_gap':('모인','벌어진'),'eye_tilt':('처진','올라간'),'under_eye':('평평한','도톰한'),
    'brow_len':('짧은','긴'),'brow_eye_gap':('붙은','떨어진'),'brow_tilt':('평평한','올라간'),'glabella_w':('좁은','넓은'),
    'nose_len':('짧은','긴'),'ala_w':('좁은','넓은'),'nose_taper':('뭉툭한','갸름한'),
    'mouth_w':('작은','큰'),'lip_up':('얇은','두꺼운'),'lip_low':('얇은','두꺼운'),'mouth_tilt':('처진','올라간'),'philtrum':('짧은','긴'),
    'symmetry':('고른','치우친'),
    'zone_top':('짧은','긴'),'zone_mid':('짧은','긴'),'zone_bot':('짧은','긴'),
    'eye_len_l':('짧은','긴'),'eye_len_r':('짧은','긴'),'eye_h_l':('가는','큰'),'eye_h_r':('가는','큰'),
    'brow_len_l':('짧은','긴'),'brow_len_r':('짧은','긴'),'jaw_angle_l':('각진','둥근'),'jaw_angle_r':('각진','둥근'),
    'upper_cheek_ratio':('이마가 좁은','이마가 넓은'),'jaw_cheek_ratio':('턱이 좁은','턱이 넓은'),'chin_jaw_ratio':('뾰족한','넓적한'),'jaw_taper':('완만한','급한'),
    'lip_h':('얇은','두꺼운'),'mouth_hw':('가로로 긴','도톰한'),
}


def _d(a, b):
    return float(np.linalg.norm(a - b))


def _angle(v1, v2):
    c = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _tilt_deg(inner, outer):
    """안쪽→바깥쪽 선의 기울기(°). 화면 y는 아래가 +이므로 부호를 뒤집어 '올라감'을 +로"""
    return math.degrees(math.atan2(-(outer[1] - inner[1]), abs(outer[0] - inner[0])))


def landmarks_to_array(landmarks, w, h):
    """(호환용) MediaPipe landmark 목록 → (N,3) 픽셀 좌표. FaceReading_3 에서는 쓰지 않음"""
    return np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks], dtype=np.float32)


def head_pose(pts):
    """정면 여부. yaw: 코끝이 두 눈 중심에서 좌우로 얼마나 벗어났나(눈 사이 거리 대비, -1~1),
       roll: 두 눈을 잇는 선의 기울기(°)."""
    le = (pts[P['eye_l_out']] + pts[P['eye_l_in']]) / 2
    re = (pts[P['eye_r_in']] + pts[P['eye_r_out']]) / 2
    mid = (le + re) / 2
    eye_dist = _d(le[:2], re[:2]) + 1e-6
    yaw = float((pts[P['nose_tip']][0] - mid[0]) / eye_dist)
    roll = math.degrees(math.atan2(re[1] - le[1], re[0] - le[0]))
    return yaw, roll


def compute_metrics(pts, hairline=None):
    """pts: (103,3) 또는 (98,2|3) 픽셀 좌표. 반환: {이름: 값} (METRIC_NAMES 순서)
    hairline: hairline.hairline_from_mask() 결과(dict) 또는 None.
              있으면 이마 상단(10번 점 대신 실제 헤어라인)과 이마 폭(피부 기준)을 그 값으로 잰다."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) == 98:                             # 가상 점이 없으면 붙인다
        pts = add_virtual_points(pts).astype(float)
    if hairline is not None:                       # 10번 점을 헤어라인 중앙으로 바꾼 복사본 사용
        pts = pts.copy()
        pts[P['forehead_top']][0] = hairline['top'][0]; pts[P['forehead_top']][1] = hairline['top'][1]
    g = lambda k: pts[P[k]]
    # 기준 길이: 눈동자(96,97) 거리 (가상 점까지 103개가 정상. 98개만 오면 눈 중심 거리)
    if len(pts) >= 98:
        ipd = _d(g('iris_l')[:2], g('iris_r')[:2])
    else:
        le = (g('eye_l_out') + g('eye_l_in')) / 2; re = (g('eye_r_in') + g('eye_r_out')) / 2
        ipd = _d(le[:2], re[:2])
    ipd = max(ipd, 1e-6)
    n = lambda a, b: _d(a[:2], b[:2]) / ipd

    face_len = n(g('forehead_top'), g('chin'))
    cheek_w = n(g('cheek_l'), g('cheek_r'))
    m = {}
    # 윤곽
    m['face_ratio'] = face_len / max(cheek_w, 1e-6)
    m['forehead_w'] = n(g('temple_l'), g('temple_r')) if hairline is None else \
                      _d(np.array(hairline['left'], float), np.array(hairline['right'], float)) / ipd
    m['cheek_w'] = cheek_w
    m['jaw_w'] = n(g('jaw_l'), g('jaw_r'))
    m['chin_w'] = n(g('chin_l'), g('chin_r'))
    m['chin_len'] = n(g('lip_bot'), g('chin'))
    m['jaw_angle_l'] = _angle(g('cheek_l')[:2] - g('jaw_l')[:2], g('chin')[:2] - g('jaw_l')[:2])
    m['jaw_angle_r'] = _angle(g('cheek_r')[:2] - g('jaw_r')[:2], g('chin')[:2] - g('jaw_r')[:2])
    m['jaw_angle'] = (m['jaw_angle_l'] + m['jaw_angle_r']) / 2
    # 윤곽 비율 (팀 측정표)
    m['upper_cheek_ratio'] = m['forehead_w'] / max(cheek_w, 1e-6)        # 상안부폭 / 광대폭
    m['jaw_cheek_ratio']   = m['jaw_w'] / max(cheek_w, 1e-6)             # 턱폭 / 광대폭
    m['chin_jaw_ratio']    = m['chin_w'] / max(m['jaw_w'], 1e-6)         # 턱끝폭 / 턱폭
    m['jaw_taper']         = (cheek_w - m['chin_w']) / max(cheek_w, 1e-6)  # 턱 좁아짐: 광대에서 턱끝까지 폭이 줄어드는 비율 (0=안 줄어듦)
    # 눈
    m['eye_len_l'] = n(g('eye_l_out'), g('eye_l_in')); m['eye_len_r'] = n(g('eye_r_in'), g('eye_r_out'))
    m['eye_len'] = (m['eye_len_l'] + m['eye_len_r']) / 2
    m['eye_h_l'] = n(g('eye_l_top'), g('eye_l_bot')); m['eye_h_r'] = n(g('eye_r_top'), g('eye_r_bot'))
    eh = (m['eye_h_l'] + m['eye_h_r']) / 2
    m['eye_aspect'] = m['eye_len'] / max(eh, 1e-6)
    m['eye_gap'] = n(g('eye_l_in'), g('eye_r_in'))
    m['eye_tilt'] = (_tilt_deg(g('eye_l_in')[:2], g('eye_l_out')[:2]) + _tilt_deg(g('eye_r_in')[:2], g('eye_r_out')[:2])) / 2
    m['under_eye'] = (n(g('eye_l_bot'), g('under_eye_l')) + n(g('eye_r_bot'), g('under_eye_r'))) / 2
    # 눈썹
    m['brow_len_l'] = n(g('brow_l_out'), g('brow_l_in')); m['brow_len_r'] = n(g('brow_r_in'), g('brow_r_out'))
    m['brow_len'] = (m['brow_len_l'] + m['brow_len_r']) / 2
    bl = (g('brow_l_out') + g('brow_l_in')) / 2; br = (g('brow_r_out') + g('brow_r_in')) / 2
    m['brow_eye_gap'] = (n(bl, g('eye_l_top')) + n(br, g('eye_r_top'))) / 2
    m['brow_tilt'] = (_tilt_deg(g('brow_l_in')[:2], g('brow_l_out')[:2]) + _tilt_deg(g('brow_r_in')[:2], g('brow_r_out')[:2])) / 2
    m['glabella_w'] = n(g('brow_l_in'), g('brow_r_in'))
    # 코
    m['nose_len'] = n(g('nose_bridge'), g('subnasale'))
    m['ala_w'] = n(g('ala_l'), g('ala_r'))
    m['nose_taper'] = m['nose_len'] / max(m['ala_w'], 1e-6)
    # 입
    m['mouth_w'] = n(g('mouth_l'), g('mouth_r'))
    m['lip_up'] = n(g('lip_top'), g('lip_up_bot'))
    m['lip_low'] = n(g('lip_low_top'), g('lip_bot'))
    mc = (g('lip_top') + g('lip_bot')) / 2
    m['mouth_tilt'] = (_tilt_deg(mc[:2], g('mouth_l')[:2]) + _tilt_deg(mc[:2], g('mouth_r')[:2])) / 2
    m['philtrum'] = n(g('subnasale'), g('lip_top'))
    m['lip_h'] = n(g('lip_top'), g('lip_bot'))                            # 입술 전체 두께 (윗입술 위~아랫입술 아래)
    m['mouth_hw'] = m['lip_h'] / max(m['mouth_w'], 1e-6)                  # 입 세로/가로
    # 대칭: 얼굴 중심선(이마 위~턱끝)에 대해 왼쪽 점을 반사한 위치와 오른쪽 점의 거리
    top, bot = g('forehead_top')[:2], g('chin')[:2]
    axis = (bot - top) / (np.linalg.norm(bot - top) + 1e-9)
    errs = []
    for l, r in SYM_PAIRS:
        pl, pr = pts[l][:2], pts[r][:2]
        v = pl - top
        refl = top + 2 * np.dot(v, axis) * axis - v          # 축 대칭
        errs.append(np.linalg.norm(refl - pr) / ipd)
    m['symmetry'] = float(np.mean(errs))
    # 세 구역
    z1 = n(g('forehead_top'), g('glabella')); z2 = n(g('glabella'), g('subnasale')); z3 = n(g('subnasale'), g('chin'))
    tot = z1 + z2 + z3 + 1e-9
    m['zone_top'], m['zone_mid'], m['zone_bot'] = 100 * z1 / tot, 100 * z2 / tot, 100 * z3 / tot
    return {k: float(m[k]) for k in METRIC_NAMES}


def metrics_vector(m):
    return np.array([m[k] for k in METRIC_NAMES], dtype=np.float32)


def percentile_of(value, sorted_ref):
    """기준 분포(오름차순 배열)에서 value의 백분위(0~100)"""
    sorted_ref = np.asarray(sorted_ref)
    return float(np.searchsorted(sorted_ref, value, side='right') / len(sorted_ref) * 100)


def level_label(name, pct):
    """백분위 → 5단계 한글 라벨. 하위 10% / 10~30% / 30~70% / 70~90% / 상위 10%"""
    lo, hi = LEVEL_WORDS.get(name, ('작은', '큰'))
    if pct < 10:  return 0, LEVELS[0] % lo
    if pct < 30:  return 1, LEVELS[1] % lo
    if pct < 70:  return 2, LEVELS[2]
    if pct < 90:  return 3, LEVELS[3] % hi
    return 4, LEVELS[4] % hi


def geometric_face_shape(m):
    """MLP가 없을 때 쓰는 규칙 기반 얼굴형 (0 heart, 1 oblong, 2 oval, 3 round, 4 square)"""
    r, fw, cw, jw, ang = m['face_ratio'], m['forehead_w'], m['cheek_w'], m['jaw_w'], m['jaw_angle']
    if r > 1.55:                       return 1          # oblong
    if fw > jw * 1.12 and r > 1.3:     return 0          # heart
    if ang < 118 and jw > cw * 0.82:   return 4          # square
    if r < 1.3:                        return 3          # round
    return 2                                             # oval


FACE_SHAPES = ['heart', 'oblong', 'oval', 'round', 'square']


# =============================================================================
#  [3] hairline - 헤어라인 분할 (선택)
# =============================================================================

# =====================================================================
#  hairline  -  우리가 학습한 분할 모델(hair_seg.tflite, CelebAMask-HQ 4클래스)로 헤어라인(진짜 이마 상단) 찾기
#
#  WFLW 98점에는 이마 위 점이 없어서 가상 점(미간 + 코 길이)으로 어림잡는데, 분할 모델이 있으면
#  픽셀마다 0 배경 / 1 얼굴 피부 / 2 머리카락 / 3 모자 를 붙여 주므로 피부가 위로 끝나는 곳을 헤어라인으로 쓴다.
#
#  모델: hair_seg.tflite (HairSeg_Train.ipynb 결과, 약 2.3MB, 같은 폴더). 실행기는 wflw_mesh 와 같은 LiteRT.
#  파일이 없으면 Segmenter.ok == False 가 되고 가상 이마 점을 그대로 쓴다.
# =====================================================================

MODEL_FILE = 'hair_seg.tflite'
FACE_SKIN, HAIR, HAT = 1, 2, 3          # hair_seg.tflite 의 클래스 번호 (HairSeg_Train.ipynb CLASSES 와 같아야 함)
SEG_SIZE = 256                          # 학습 때 입력 크기 (HairSeg_Train.ipynb 의 S)


class Segmenter:
    """hair_seg.tflite 를 LiteRT 로 직접 실행. rgb (H,W,3) → (H,W) uint8 클래스 맵"""

    def __init__(self, path=MODEL_FILE):
        self.ok = False
        path = model_path(path)
        if not os.path.exists(path):
            print('%s 없음 → 헤어라인 미사용 (가상 이마 점 사용)' % path); return
        try:
            self.it = _interpreter(path)                      # [4] wflw_mesh 구역의 함수 (ai_edge_litert → tflite_runtime → tensorflow)
            self.inp = self.it.get_input_details()[0]
            self.out = self.it.get_output_details()[0]
            self.ok = True; print('헤어라인 분할 모델 로딩 완료:', path)
        except Exception as e:
            print('분할 모델 로딩 실패 → 헤어라인 미사용:', e)

    def mask(self, rgb):
        if not self.ok: return None
        h, w = rgb.shape[:2]
        x = cv2.resize(rgb, (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)[None] / 255.0
        self.it.set_tensor(self.inp['index'], x); self.it.invoke()
        m = self.it.get_tensor(self.out['index'])[0].argmax(-1).astype(np.uint8)
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)


def hairline_from_mask(mask, pts, P):
    """
    mask: (H,W) 카테고리 마스크, pts: Face Mesh (468+,3) 픽셀 좌표, P: face_metrics.P
    반환: dict(top=(x,y) 이마 상단 중앙, left=(x,y), right=(x,y) 이마 폭 양끝, ys=열별 헤어라인 y) 또는 None
    """
    H, W = mask.shape[:2]
    gl = pts[P['glabella']]; bl, br = pts[P['brow_l_out']], pts[P['brow_r_out']]
    x0, x1 = int(max(min(bl[0], br[0]), 0)), int(min(max(bl[0], br[0]), W - 1))
    brow_y = int(min(bl[1], br[1], gl[1]))
    if x1 - x0 < 10 or brow_y < 5: return None
    skin = (mask == FACE_SKIN)
    # 열마다: 눈썹 높이에서 위로 올라가며 얼굴 피부가 끊기는 첫 y (=헤어라인)
    band = skin[:brow_y, x0:x1 + 1]                       # (brow_y, ncols), 아래쪽이 눈썹
    ys = np.full(band.shape[1], -1, int)
    for i in range(band.shape[1]):
        col = band[:, i]
        if not col[-3:].any(): continue                   # 눈썹 바로 위가 피부가 아니면(안경·머리카락) 건너뜀
        nz = np.where(~col)[0]                            # 피부가 아닌 행들
        nz = nz[nz < brow_y - 3]
        ys[i] = (nz[-1] + 1) if len(nz) else 0             # 가장 아래쪽 비피부 행 바로 밑 = 피부 시작
    valid = ys >= 0
    if valid.sum() < 0.3 * len(ys): return None
    cols = np.arange(x0, x1 + 1)
    # 중앙 20% 열의 중앙값 → 이마 상단
    cx = int(gl[0]); near = valid & (np.abs(cols - cx) < 0.1 * (x1 - x0) + 2)
    if near.sum() < 3: return None
    top_y = int(np.median(ys[near]))
    # 이마 폭: 눈썹과 헤어라인 사이 40% 높이에서, 중심을 포함하는 연속 피부 구간
    y_mid = int(brow_y - 0.4 * (brow_y - top_y))
    row = skin[y_mid]
    if not row[cx]: return None
    l = cx
    while l > 0 and row[l - 1]: l -= 1
    r = cx
    while r < W - 1 and row[r + 1]: r += 1
    return dict(top=(cx, top_y), left=(l, y_mid), right=(r, y_mid), ys=ys, cols=cols, brow_y=brow_y)


# =============================================================================
#  [4] wflw_mesh - YuNet + tflite 로 98점 찾기
# =============================================================================

# =====================================================================
#  wflw_mesh.py  -  우리가 학습한 WFLW 98점 모델로 얼굴 점 찾기 (Colab·보드 공용)
#
#  얼굴 검출: OpenCV YuNet (face_detection_yunet_2023mar.onnx, 같은 폴더)
#  점 찾기  : wflw_mesh.tflite (WFLW_Train.ipynb 가 만든 것, 같은 폴더)
#  반환     : (103,3) 픽셀 좌표  = WFLW 98점 + 가상 점 5개 (wflw_prep.add_virtual_points)
#
#  tflite 실행기는 tflite_runtime → ai_edge_litert → tensorflow 순서로 찾는다.
# =====================================================================

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


# =====================================================================
#  eye_prep  -  2단계 눈 정밀 모델의 크롭 규칙  (colab/eye_prep.py 와 똑같아야 한다)
#
#  1단계 wflw_mesh 가 찍은 98점은 눈·눈썹이 '평균 모양'으로 뭉개지는 경향이 있다 (GAP+Dense 회귀 구조의 한계).
#  그래서 1단계 결과로 눈+눈썹 주변만 잘라 작은 2단계 모델(eye_refine.tflite)에 다시 넣어 18점을 정밀하게 다시 찍는다.
#  오른쪽 눈은 크롭을 좌우 반전해서 '왼쪽 눈'으로 만들어 넣고, 결과를 다시 반전한다 (모델은 왼쪽 눈만 배운다).
# =====================================================================

EYE_IMG = 64            # 2단계 모델 입력 한 변
EYE_MARGIN = 1.7        # 눈+눈썹 18점이 차지하는 긴 변 × 1.7 을 정사각 한 변으로

LEFT_IDS = list(range(33, 42)) + list(range(60, 68)) + [96]                                   # 왼쪽 눈썹·눈·눈동자
RIGHT_IDS = [46, 45, 44, 43, 42, 50, 49, 48, 47] + [72, 71, 70, 69, 68, 75, 74, 73] + [97]   # FLIP_PAIRS 짝
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


def eye_pts_to_image(p01, x0, y0, s, side):
    """모델 출력 (18,2) 0~1 → 원본 픽셀 좌표 (18,2). 오른쪽이면 x 를 되돌린다."""
    p = np.asarray(p01, np.float32).reshape(N_EYE_PTS, 2).copy()
    if side == "right":
        p[:, 0] = 1.0 - p[:, 0]
    return p * s + np.array([x0, y0], np.float32)


EYE_FILE = 'eye_refine.tflite'


class EyeRefiner:
    """2단계: 1단계 98점 → 눈+눈썹 크롭 → eye_refine.tflite → 양쪽 18점씩 덮어쓰기. 모델 파일이 없으면 아무것도 안 함."""

    def __init__(self, enabled=True, path=EYE_FILE):
        path = model_path(path)
        self.ok = enabled and os.path.exists(path)
        if not enabled:
            print("[INFO] 눈 정밀 모델 끔 (--no-eye-refine)")
        elif not self.ok:
            print(f"[INFO] {EYE_FILE} 없음 → 1단계 눈 점 그대로 사용 (EyeRefine_Train.ipynb 로 만들어 같은 폴더에 두면 켜짐)")
        if self.ok:
            self.it = _interpreter(path)
            self.inp = self.it.get_input_details()[0]
            self.out = self.it.get_output_details()[0]
            print(f"[INFO] 눈 정밀 모델 사용: {path}")

    def __call__(self, bgr, pts98):
        """(98,2) 픽셀 좌표 → 눈·눈썹 18점×2 를 정밀값으로 바꾼 (98,2)"""
        if not self.ok:
            return pts98
        out = np.array(pts98, np.float32, copy=True)
        for side in ("left", "right"):
            x0, y0, s = eye_crop_box(pts98, side)
            self.it.set_tensor(self.inp['index'], eye_to_input(crop_eye(bgr, x0, y0, s, side))); self.it.invoke()
            p01 = self.it.get_tensor(self.out['index'])[0]
            out[EYE_IDS[side], :2] = eye_pts_to_image(p01, x0, y0, s, side)
        return out


class WflwMesh:
    def __init__(self, mesh=MESH_FILE, detector=DET_FILE, score_thr=0.7, eye_refine=True):
        mesh, detector, our_det = model_path(mesh), model_path(detector), model_path(OUR_DET_FILE)
        if not os.path.exists(mesh):
            raise FileNotFoundError('%s 없음. colab 결과(WFLW_Train.ipynb)를 face_fortune.py 와 같은 폴더에 두세요' % mesh)
        self.it = _interpreter(mesh)
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]
        self.refiner = EyeRefiner(enabled=eye_refine)
        self.last_box = None
        # 얼굴 검출기: 우리 YOLO(face_det_int8.tflite) 가 있으면 그것, 없으면 OpenCV YuNet
        self.yolo = None; self.det = None
        if os.path.exists(our_det):
            self.yolo = YoloFaceDetector(our_det); det_name = 'our YOLO (%s)' % our_det
        elif os.path.exists(detector):
            self.det = cv2.FaceDetectorYN.create(detector, '', (320, 320), score_threshold=score_thr)
            self.det_size = (320, 320); det_name = 'YuNet (%s)' % detector
        else:
            raise FileNotFoundError('얼굴 검출기 없음: %s (FaceDet_Train.ipynb 결과를 face_fortune.py 와 같은 폴더에) 또는 %s (%s)' % (our_det, detector, DET_URL))
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
        pts = self.landmarks(bgr, box)
        pts = self.refiner(bgr, pts)
        return add_virtual_points(pts)


# =============================================================================
#  [5] face_fortune - 촬영 · 검사 · 저장 · 화면
# =============================================================================

ROOT = Path(__file__).resolve().parent
CAPTURE_DIR = ROOT / "captures"
REFERENCE_CSV = ROOT / "face_reference_10level_clean.csv"
REFERENCE_PERCENTILES = tuple(f"p{percentile}" for percentile in range(10, 100, 10))

# 웹 키오스크 분석 중 미리보기용 파일.
# face_fortune.py가 카메라를 단독으로 사용하면서,
# 현재 분석 화면을 정적 JPEG/JSON으로 계속 갱신한다.
LIVE_PREVIEW_DIR = ROOT / "static"
LIVE_PREVIEW_IMAGE = LIVE_PREVIEW_DIR / "live_analysis.jpg"
LIVE_PREVIEW_STATE = LIVE_PREVIEW_DIR / "live_analysis.json"
LIVE_PREVIEW_INTERVAL_SEC = 0.12   # 약 8 fps
LIVE_PREVIEW_JPEG_QUALITY = 82

CAMERA_FALLBACK = (0, 1, 2)
TARGET_FRAMES = 30          # 이만큼 '좋은' 프레임을 모으면 평균을 낸다
CNN_MULTI_FRAME_SAMPLE_NUMBERS = (1, 8, 15, 22, 30)
CNN_FACE_CROP_MARGIN = 1.35
CNN_FACE_V41_BIN_RANGES = ((1, 6), (7, 12), (13, 18), (19, 24), (25, 30))
FOREHEAD_EVERY = 3          # 헤어라인 분할은 무거워서 3프레임에 한 번
RESET_AFTER_INVALID = 8     # 연속으로 이만큼 조건이 깨지면 처음부터

# 정면 조건.  yaw 는 face_metrics.head_pose 의 비율(-1~1), roll 은 도(°),
# pitch 는 2D 모델이라 각도가 없어 "눈~턱 사이에서 코끝이 어디쯤인가" 비율로 대신한다.
MAX_YAW_RATIO = 0.12
MAX_ROLL_DEG = 10.0
PITCH_PROXY_RANGE = (0.30, 0.56)   # (코끝y - 눈y) / (턱y - 눈y). 정면이면 0.4 근처
MIN_FACE_WIDTH_RATIO = 0.22        # 얼굴 폭(광대) / 화면 폭

GAZE_X = (0.40, 0.60)      # 눈동자가 눈 가로의 40~60% 사이 (렌즈를 보고 있음)
GAZE_Y = (0.30, 0.70)
BLINK_RATIO = 0.15         # 눈 세로/가로 가 이보다 작으면 감은 것

STABILITY_LIMIT = 0.045    # 모은 프레임들의 측정값 변동(CV) 평균이 이 이하면 '안정'
CENTER_JUMP_LIMIT = 0.08   # 얼굴 중심이 화면 폭의 8% 넘게 튀면 다시 모음

# WFLW 98점 번호 (사진 기준 왼쪽/오른쪽)
CONTOUR = list(range(0, 33))
EYES = {
    "left":  dict(outer=60, inner=64, top=62, bottom=66, pupil=96, ring=list(range(60, 68))),
    "right": dict(outer=72, inner=68, top=70, bottom=74, pupil=97, ring=list(range(68, 76))),
}
BROWS = {"left": list(range(33, 42)), "right": list(range(42, 51))}
MOUTH_OUTER = list(range(76, 88))
NOSE = [51, 52, 53, 54, 55, 56, 57, 58, 59]

# 화면에 그릴 측정선 (이름, 시작 점, 끝 점)
MEASURE_LINES = [
    ("face_len", P["forehead_top"], P["chin"]),
    ("cheek_w", P["cheek_l"], P["cheek_r"]),
    ("jaw_w", P["jaw_l"], P["jaw_r"]),
    ("nose_len", P["nose_bridge"], P["subnasale"]),
    ("mouth_w", P["mouth_l"], P["mouth_r"]),
]


# -----------------------------------------------------------------------------
# 점 찾기 (우리 모델) + 헤어라인 (선택)
# -----------------------------------------------------------------------------

class Landmarker:
    """frame → (103,3) 픽셀 좌표 (WFLW 98점 + 가상 5점) 또는 None"""

    def __init__(self, eye_refine: bool = True) -> None:
        self.mesh = WflwMesh(eye_refine=eye_refine)

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        return self.mesh(frame_bgr)


class Hairline:
    """우리 분할 모델(hair_seg.tflite)로 실제 이마 상단·이마 폭. 모델 없으면 ok=False → 가상 이마 점 사용.
    학습 데이터(CelebAMask-HQ)가 얼굴이 꽉 찬 구도라서, 프레임 전체가 아니라 얼굴 주변만 잘라 넣고 결과를 제자리에 되돌린다."""

    def __init__(self, enabled: bool = True) -> None:
        self.ok = False; self.last_mask = None
        if not enabled:
            print("[INFO] 헤어라인 끔 (--no-hairline)")
            return
        try:
            self.seg = Segmenter()
            self._from_mask = hairline_from_mask
            self.ok = bool(self.seg.ok)
        except Exception as error:
            print("[INFO] 헤어라인 미사용:", error)

    def __call__(self, frame_bgr: np.ndarray, pts: np.ndarray) -> dict[str, Any] | None:
        if not self.ok:
            return None
        H, W = frame_bgr.shape[:2]
        x1, y1, x2, y2 = face_bbox(pts)
        cx, cy, s = (x1 + x2) / 2.0, (y1 + y2) / 2.0, float(max(x2 - x1, y2 - y1))
        X1, Y1 = int(max(cx - s * 0.9, 0)), int(max(cy - s * 1.1, 0))       # 위쪽을 더 넉넉히 (머리카락까지)
        X2, Y2 = int(min(cx + s * 0.9, W)), int(min(cy + s * 0.9, H))
        if X2 - X1 < 32 or Y2 - Y1 < 32:
            return None
        crop = cv2.cvtColor(frame_bgr[Y1:Y2, X1:X2], cv2.COLOR_BGR2RGB)
        m = self.seg.mask(crop)
        if m is None:
            return None
        full = np.zeros((H, W), np.uint8)
        full[Y1:Y2, X1:X2] = m                                                # 원래 프레임 좌표로 복원
        self.last_mask = full                                                 # simple 호환 이마 지표(simple_forehead)에서 사용
        return self._from_mask(full, pts, P)


# -----------------------------------------------------------------------------
# 프레임 단위 분석
# -----------------------------------------------------------------------------

def face_bbox(pts: np.ndarray) -> tuple[int, int, int, int]:
    c = pts[CONTOUR, :2]
    top = pts[P["forehead_top"], 1]
    return int(c[:, 0].min()), int(min(top, c[:, 1].min())), int(c[:, 0].max()), int(c[:, 1].max())


def cnn_face_crop(frame_bgr: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Create a consistent square face crop for the animal CNN candidates."""
    x1, y1, x2, y2 = face_bbox(pts)
    x0, y0, side = crop_box(x1, y1, x2, y2, margin=CNN_FACE_CROP_MARGIN)
    return crop_face(frame_bgr, x0, y0, side)


def cnn_face_sharpness(face_crop: np.ndarray) -> float:
    """Laplacian variance used only to reject relatively blurry V4.1 crops."""
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_v41_face_crops(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pick one sharp crop from each of five temporal bins across the 30 accepted frames."""
    selected: list[dict[str, Any]] = []
    for start, end in CNN_FACE_V41_BIN_RANGES:
        rows = [
            item for item in candidates
            if start <= int(item["sample_number"]) <= end
        ]
        if not rows:
            continue
        best = max(rows, key=lambda item: float(item["sharpness"]))
        selected.append(best)
    return selected


def face_center(pts: np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = face_bbox(pts)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def pose_of(pts: np.ndarray) -> dict[str, float]:
    yaw, roll = head_pose(pts)
    eye_y = (pts[EYES["left"]["pupil"], 1] + pts[EYES["right"]["pupil"], 1]) / 2.0
    chin_y = pts[P["chin"], 1]
    nose_y = pts[P["nose_tip"], 1]
    pitch_proxy = float((nose_y - eye_y) / max(chin_y - eye_y, 1e-6))
    return {"yaw_ratio": float(yaw), "roll_deg": float(roll), "pitch_proxy": pitch_proxy}


def is_frontal(pose: dict[str, float]) -> bool:
    return (abs(pose["yaw_ratio"]) <= MAX_YAW_RATIO
            and abs(pose["roll_deg"]) <= MAX_ROLL_DEG
            and PITCH_PROXY_RANGE[0] <= pose["pitch_proxy"] <= PITCH_PROXY_RANGE[1])


def eye_metrics(pts: np.ndarray, side: str) -> dict[str, float]:
    e = EYES[side]
    outer, inner = pts[e["outer"], :2], pts[e["inner"], :2]
    top, bottom, pupil = pts[e["top"], :2], pts[e["bottom"], :2], pts[e["pupil"], :2]
    eye_w = float(np.linalg.norm(inner - outer)) + 1e-9
    eye_h = float(np.linalg.norm(bottom - top))
    axis = inner - outer
    gaze_x = float(np.dot(pupil - outer, axis) / (np.dot(axis, axis) + 1e-9))
    gaze_y = float((pupil[1] - top[1]) / ((bottom[1] - top[1]) + 1e-9))
    tilt = math.degrees(math.atan2(-(outer[1] - inner[1]), abs(outer[0] - inner[0])))
    return {
        "opening_width_ratio": eye_h / eye_w,
        "outer_corner_tilt_deg": float(tilt),
        "gaze_x": gaze_x,
        "gaze_y": gaze_y,
    }


def eyes_open(eyes: dict[str, dict[str, float]]) -> bool:
    return min(e["opening_width_ratio"] for e in eyes.values()) >= BLINK_RATIO


def gaze_centered(eyes: dict[str, dict[str, float]]) -> bool:
    return all(GAZE_X[0] <= e["gaze_x"] <= GAZE_X[1] and GAZE_Y[0] <= e["gaze_y"] <= GAZE_Y[1]
               for e in eyes.values())



# -----------------------------------------------------------------------------
# simple/face_fortune.py 호환 측정값  (이름·공식 동일, MediaPipe 점 → WFLW 점으로만 바꿈)
#   simple 의 "right" = 사진 왼쪽 눈(MediaPipe 33/133) = WFLW 60~67·96,  "left" = 사진 오른쪽 = WFLW 68~75·97
#   MediaPipe 10번(이마 위)·54/284(관자놀이)는 WFLW 에 없어 헤어라인(있으면) 또는 가상 점 98/99/100 으로 대신한다
# -----------------------------------------------------------------------------
SIMPLE_EYES = {
    "right": dict(outer=60, inner=64, top=62, bottom=66, iris=96),     # 사진 왼쪽
    "left":  dict(outer=72, inner=68, top=70, bottom=74, iris=97),     # 사진 오른쪽
}
SIMPLE_BROW = {"right": [33, 34, 35, 36, 37], "left": [46, 45, 44, 43, 42]}   # 바깥 → 안쪽 (MediaPipe 70…107 / 300…336 순서와 같게)
SIMPLE_FEATURE_NAMES = [
    "face_hw_ratio", "upper_face_width_ratio", "right_eye_width_ratio", "left_eye_width_ratio",
    "right_eye_height_ratio", "left_eye_height_ratio", "eye_distance_ratio", "eye_hw_ratio",
    "right_eyebrow_length_ratio", "left_eyebrow_length_ratio", "eyebrow_length_ratio",
    "brow_eye_distance_ratio", "brow_gap_ratio", "nose_width_ratio", "nose_length_ratio", "nose_hw_ratio",
    "philtrum_ratio", "mouth_width_ratio", "lip_height_ratio", "mouth_hw_ratio", "cheek_width_ratio",
    "jaw_width_ratio", "chin_width_ratio", "chin_length_ratio", "left_jaw_angle_deg", "right_jaw_angle_deg",
    "jaw_angle_deg", "upper_cheek_ratio", "jaw_cheek_ratio", "chin_jaw_ratio", "jaw_taper",
    "cheek_to_jaw_width_ratio", "cheek_to_upper_face_width_ratio", "upper_face_ratio", "middle_face_ratio", "lower_face_ratio",
]


def _angle3(a, b, c) -> float:
    v1, v2 = a - b, c - b
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _polyline(pts, ids) -> float:
    q = pts[ids, :2]
    return float(np.sum(np.linalg.norm(q[1:] - q[:-1], axis=1)))


def simple_features(pts: np.ndarray, hairline: dict[str, Any] | None) -> tuple[dict[str, float], dict[str, tuple]]:
    """simple/face_fortune.py 의 calculate_features() 와 같은 36개 비율. 반환: (features, 측정선)"""
    q = lambda i: pts[i, :2].astype(np.float64)
    face_top = np.array(hairline["top"], np.float64) if hairline is not None else q(98)
    chin = q(16); left_cheek, right_cheek = q(0), q(32)
    if hairline is not None:
        left_upper, right_upper = np.array(hairline["left"], np.float64), np.array(hairline["right"], np.float64)
    else:
        left_upper, right_upper = q(99), q(100)
    left_jaw, right_jaw = q(5), q(27); left_chin, right_chin = q(13), q(19)

    face_width = np.linalg.norm(right_cheek - left_cheek) + 1e-9
    face_height = np.linalg.norm(chin - face_top) + 1e-9
    upper_face_width = np.linalg.norm(right_upper - left_upper)
    cheek_width = face_width
    jaw_width = np.linalg.norm(right_jaw - left_jaw)
    chin_width = np.linalg.norm(right_chin - left_chin)

    R, L = SIMPLE_EYES["right"], SIMPLE_EYES["left"]
    right_eye_outer, right_eye_inner, left_eye_inner, left_eye_outer = q(R["outer"]), q(R["inner"]), q(L["inner"]), q(L["outer"])
    right_eye_width = np.linalg.norm(right_eye_inner - right_eye_outer); left_eye_width = np.linalg.norm(left_eye_inner - left_eye_outer)
    right_eye_height = np.linalg.norm(q(R["bottom"]) - q(R["top"])); left_eye_height = np.linalg.norm(q(L["bottom"]) - q(L["top"]))
    eye_distance = np.linalg.norm(left_eye_inner - right_eye_inner)

    eyebrow_length_right = _polyline(pts, SIMPLE_BROW["right"]); eyebrow_length_left = _polyline(pts, SIMPLE_BROW["left"])
    eyebrow_length = (eyebrow_length_right + eyebrow_length_left) * 0.5
    right_brow_center = pts[SIMPLE_BROW["right"], :2].mean(0); left_brow_center = pts[SIMPLE_BROW["left"], :2].mean(0)
    right_eye_center = (right_eye_outer + right_eye_inner) / 2; left_eye_center = (left_eye_outer + left_eye_inner) / 2
    brow_eye_distance = (np.linalg.norm(right_brow_center - right_eye_center) + np.linalg.norm(left_brow_center - left_eye_center)) * 0.5
    brow_gap = np.linalg.norm(q(37) - q(42))

    nose_left, nose_right, nose_top, nose_bottom = q(55), q(59), q(51), q(57)
    nose_width = np.linalg.norm(nose_right - nose_left); nose_length = np.linalg.norm(nose_bottom - nose_top)
    mouth_left, mouth_right, lip_top, lip_bottom = q(76), q(82), q(79), q(85)
    mouth_width = np.linalg.norm(mouth_right - mouth_left); lip_height = np.linalg.norm(lip_bottom - lip_top)
    philtrum_length = np.linalg.norm(lip_top - nose_bottom)
    chin_length = np.linalg.norm(chin - lip_bottom)

    left_jaw_angle = _angle3(q(2), q(5), q(9))          # MediaPipe 132-172-149 에 해당 (턱선 위 이웃 - 턱 각 - 턱끝 쪽 이웃)
    right_jaw_angle = _angle3(q(30), q(27), q(23))      # MediaPipe 288-397-378
    jaw_angle = (left_jaw_angle + right_jaw_angle) * 0.5

    brow_mid = (right_brow_center + left_brow_center) / 2; nose_base = nose_bottom
    upper_h = max(0.0, brow_mid[1] - face_top[1]); middle_h = max(0.0, nose_base[1] - brow_mid[1]); lower_h = max(0.0, chin[1] - nose_base[1])
    vertical_total = upper_h + middle_h + lower_h + 1e-9
    avg_eye_width = (right_eye_width + left_eye_width) * 0.5 + 1e-9; avg_eye_height = (right_eye_height + left_eye_height) * 0.5

    f = {
        "face_hw_ratio": face_height / face_width,
        "upper_face_width_ratio": upper_face_width / face_width,
        "right_eye_width_ratio": right_eye_width / face_width, "left_eye_width_ratio": left_eye_width / face_width,
        "right_eye_height_ratio": right_eye_height / face_width, "left_eye_height_ratio": left_eye_height / face_width,
        "eye_distance_ratio": eye_distance / face_width, "eye_hw_ratio": avg_eye_height / avg_eye_width,
        "right_eyebrow_length_ratio": eyebrow_length_right / face_width, "left_eyebrow_length_ratio": eyebrow_length_left / face_width,
        "eyebrow_length_ratio": eyebrow_length / face_width,
        "brow_eye_distance_ratio": brow_eye_distance / face_height, "brow_gap_ratio": brow_gap / face_width,
        "nose_width_ratio": nose_width / face_width, "nose_length_ratio": nose_length / face_height, "nose_hw_ratio": nose_length / (nose_width + 1e-9),
        "philtrum_ratio": philtrum_length / face_height,
        "mouth_width_ratio": mouth_width / face_width, "lip_height_ratio": lip_height / face_height, "mouth_hw_ratio": lip_height / (mouth_width + 1e-9),
        "cheek_width_ratio": 1.0, "jaw_width_ratio": jaw_width / face_width, "chin_width_ratio": chin_width / face_width,
        "chin_length_ratio": chin_length / face_height,
        "left_jaw_angle_deg": left_jaw_angle, "right_jaw_angle_deg": right_jaw_angle, "jaw_angle_deg": jaw_angle,
        "upper_cheek_ratio": upper_face_width / cheek_width, "jaw_cheek_ratio": jaw_width / cheek_width,
        "chin_jaw_ratio": chin_width / (jaw_width + 1e-9), "jaw_taper": 1.0 - chin_width / (jaw_width + 1e-9),
        "cheek_to_jaw_width_ratio": cheek_width / (jaw_width + 1e-9), "cheek_to_upper_face_width_ratio": cheek_width / (upper_face_width + 1e-9),
        "upper_face_ratio": upper_h / vertical_total, "middle_face_ratio": middle_h / vertical_total, "lower_face_ratio": lower_h / vertical_total,
    }
    lines = {"face_height": (face_top, chin), "cheek_width": (left_cheek, right_cheek), "upper_width": (left_upper, right_upper),
             "jaw_width": (left_jaw, right_jaw), "chin_width": (left_chin, right_chin), "nose_width": (nose_left, nose_right),
             "nose_length": (nose_top, nose_bottom), "mouth_width": (mouth_left, mouth_right), "eye_gap": (right_eye_inner, left_eye_inner)}
    return {k: float(f[k]) for k in SIMPLE_FEATURE_NAMES}, lines


def simple_eye(pts: np.ndarray, side: str) -> dict[str, Any]:
    """simple 의 analyze_eye() 와 동일"""
    e = SIMPLE_EYES[side]
    outer, inner, top, bottom, iris = (pts[e[k], :2].astype(np.float64) for k in ("outer", "inner", "top", "bottom", "iris"))
    eye_width = np.linalg.norm(inner - outer) + 1e-9; opening = np.linalg.norm(bottom - top); iris_diameter = eye_width * 0.30 + 1e-9
    axis = inner - outer; angle = math.degrees(math.atan2(axis[1], axis[0]))
    if side == "left": angle = -angle
    upper_clearance = abs(iris[1] - top[1]); lower_clearance = abs(bottom[1] - iris[1])
    opening_ratio = float(opening / eye_width)
    candidate = "open_eye" if opening_ratio >= 0.34 else "narrow_eye" if opening_ratio <= 0.22 else "balanced_eye"
    return {"metrics": {"opening_width_ratio": opening_ratio, "outer_corner_tilt_deg": float(angle),
                        "upper_clearance_iris_ratio": float(upper_clearance / iris_diameter),
                        "lower_clearance_iris_ratio": float(lower_clearance / iris_diameter)},
            "candidate": candidate, "candidate_validated": False, "sclera_status": "geometry_only",
            "reason": "geometry only; not verified sclera segmentation or calibrated classification"}


def simple_unavailable(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "metrics": None, "reason": reason, "boundary_points_px": [], "brow_center_px": [], "confidence": 0.0}


def simple_forehead(mask: np.ndarray, pts: np.ndarray) -> dict[str, Any]:
    """simple 의 ForeheadSegmenter.analyze() 와 같은 알고리즘 (얼굴 피부 클래스만 우리 모델 번호 FACE_SKIN 으로)"""
    height, width = mask.shape[:2]
    contour = pts[0:33, :2]
    x_min, x_max = int(max(0, np.floor(contour[:, 0].min()))), int(min(width - 1, np.ceil(contour[:, 0].max())))
    y_top, y_bottom = int(max(0, np.floor(min(contour[:, 1].min(), pts[98, 1])))), int(min(height - 1, np.ceil(contour[:, 1].max())))
    right_brow = pts[SIMPLE_BROW["right"], :2].mean(0); left_brow = pts[SIMPLE_BROW["left"], :2].mean(0)
    brow_center = (right_brow + left_brow) / 2; brow_y = int(np.clip(brow_center[1], 0, height - 1))
    face_width = max(1.0, float(x_max - x_min)); face_height = max(1.0, float(y_bottom - y_top))
    roi_left = int(np.clip(x_min + 0.16 * face_width, 0, width - 1)); roi_right = int(np.clip(x_max - 0.16 * face_width, 0, width - 1))
    roi_top = int(np.clip(y_top - 0.04 * face_height, 0, height - 1)); roi_bottom = int(np.clip(brow_y, 0, height - 1))
    if roi_right <= roi_left or roi_bottom <= roi_top:
        return simple_unavailable("invalid_forehead_roi")
    xs = np.linspace(roi_left, roi_right, 35).astype(int); boundary = []
    for x in xs:
        column = mask[roi_top:roi_bottom + 1, x]
        face_skin = np.flatnonzero(column == FACE_SKIN)
        if len(face_skin) == 0: continue
        y = roi_top + int(face_skin[0])
        if y <= brow_y: boundary.append([int(x), int(y)])
    if len(boundary) < 8:
        return simple_unavailable("insufficient_visible_forehead_boundary")
    b = np.asarray(boundary, np.float64); median_y = float(np.median(b[:, 1]))
    visible_height = max(0.0, float(brow_center[1] - median_y)); visible_width = float(b[:, 0].max() - b[:, 0].min())
    return {"status": "visible_boundary_detected",
            "metrics": {"visible_forehead_height_face_height_ratio": float(visible_height / face_height),
                        "visible_forehead_width_face_width_ratio": float(visible_width / face_width)},
            "reason": "visible face-skin boundary from our hair_seg segmentation; not a verified anatomical hairline",
            "boundary_points_px": b.tolist(), "brow_center_px": brow_center.tolist(), "confidence": float(len(b) / len(xs))}


def aggregate_simple_forehead(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    """simple 의 aggregate_forehead() 와 동일"""
    valid = [a.get("forehead_simple") for a in analyses if a.get("forehead_simple") and a["forehead_simple"].get("metrics") is not None]
    if not valid:
        return simple_unavailable("no_valid_forehead_measurements")
    keys = set.intersection(*(set(v["metrics"].keys()) for v in valid))
    best = max(valid, key=lambda item: item.get("confidence", 0.0))
    return {"status": "visible_boundary_detected",
            "metrics": {k: float(np.median([v["metrics"][k] for v in valid])) for k in keys},
            "reason": best.get("reason", "visible boundary only"),
            "boundary_points_px": best.get("boundary_points_px", []), "brow_center_px": best.get("brow_center_px", []),
            "confidence": float(np.mean([v.get("confidence", 0.0) for v in valid])),
            "valid_frame_fraction": float(len(valid) / max(1, len(analyses)))}


def aggregate_simple_eyes(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    """simple 의 aggregate_eyes() 와 동일"""
    from collections import Counter
    result: dict[str, Any] = {}
    for side in SIMPLE_EYES:
        rows = [a["eyes_simple"][side] for a in analyses]
        keys = set().union(*(row.get("metrics", {}).keys() for row in rows))
        metrics, metric_std = {}, {}
        for key in keys:
            values = [row["metrics"][key] for row in rows if key in row.get("metrics", {})]
            if len(values) >= len(rows) * 0.8:
                metrics[key] = float(np.mean(values)); metric_std[key] = float(np.std(values))
        counts = Counter(row["candidate"] for row in rows)
        candidate, count = counts.most_common(1)[0] if counts else ("unassessed", 0)
        consistent = count >= len(rows) * 0.8
        result[side] = {"metrics": metrics, "metric_std": metric_std, "candidate": candidate if consistent else "unassessed",
                        "candidate_validated": False, "geometry_valid_frame_fraction": 1.0,
                        "candidate_support_fraction": count / max(1, len(rows)), "candidate_counts": dict(counts),
                        "reason": "geometry only; not verified sclera segmentation or calibrated classification"}
    return result


def analyze_frame(pts: np.ndarray, hairline: dict[str, Any] | None, mask: np.ndarray | None = None) -> dict[str, Any]:
    eyes = {side: eye_metrics(pts, side) for side in EYES}
    forehead = None
    if hairline is not None:
        forehead = {
            "status": "hairline_segmentation",
            "top_px": [float(hairline["top"][0]), float(hairline["top"][1])],
            "left_px": [float(hairline["left"][0]), float(hairline["left"][1])],
            "right_px": [float(hairline["right"][0]), float(hairline["right"][1])],
        }
    simple_vals, simple_lines = simple_features(pts, hairline)
    return {
        "feature_values": simple_vals,                                  # simple/face_fortune.py 와 같은 36개 (이름·공식 동일)
        "feature_values_ipd": compute_metrics(pts, hairline),           # FaceReading 체계 42개 (눈동자 거리 기준)
        "eyes_simple": {side: simple_eye(pts, side) for side in SIMPLE_EYES},
        "eyes": eyes,
        "pose": pose_of(pts),
        "forehead": forehead,
        "forehead_simple": simple_forehead(mask, pts) if mask is not None else simple_unavailable("not_measured"),
        "lines": simple_lines,
    }


# -----------------------------------------------------------------------------
# 30프레임 모아서 평균 (simple/face_fortune.py 와 같은 방식)
# -----------------------------------------------------------------------------

def trimmed_mean_std(values: list[float], trim_fraction: float = 0.10) -> tuple[float, float]:
    arr = np.sort(np.asarray(values, dtype=np.float64))
    trim = int(len(arr) * trim_fraction)
    if trim > 0 and len(arr) > trim * 2:
        arr = arr[trim:-trim]
    return float(np.mean(arr)), float(np.std(arr))


def _is_angle(key: str) -> bool:
    return key.endswith("_deg") or "angle" in key or "tilt" in key


class Measurement:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.samples: list[dict[str, float]] = []
        self.centers: list[tuple[float, float]] = []
        self.mean: dict[str, float] = {}
        self.std: dict[str, float] = {}
        self.score = float("inf")
        self.stable = False

    def observe(self, feature_values: dict[str, float], center: tuple[float, float], frame_width: int) -> bool:
        if self.centers:
            jump = np.linalg.norm(np.asarray(center) - np.asarray(self.centers[-1])) / max(1, frame_width)
            if jump > CENTER_JUMP_LIMIT:
                self.reset()
        if len(self.samples) >= TARGET_FRAMES:
            return False

        self.samples.append(dict(feature_values))
        self.centers.append(center)
        if len(self.samples) < TARGET_FRAMES:
            return False

        mean, std, scores = {}, {}, []
        for key in self.samples[0].keys():
            m, s = trimmed_mean_std([float(sample[key]) for sample in self.samples])
            mean[key], std[key] = m, s
            if _is_angle(key):
                scores.append(s / 180.0)
            elif abs(m) > 1e-6:
                scores.append(abs(s / m))
        self.mean, self.std = mean, std
        self.score = float(np.mean(scores)) if scores else float("inf")
        self.stable = bool(np.isfinite(self.score) and self.score <= STABILITY_LIMIT)
        return True


def aggregate_eyes(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for side in EYES:
        rows = [a["eyes"][side] for a in analyses]
        keys = rows[0].keys()
        out[side] = {
            "metrics": {k: float(np.mean([r[k] for r in rows])) for k in keys},
            "metric_std": {k: float(np.std([r[k] for r in rows])) for k in keys},
        }
    return out


def aggregate_pose(analyses: list[dict[str, Any]]) -> dict[str, float]:
    keys = analyses[0]["pose"].keys()
    return {k: float(np.mean([a["pose"][k] for a in analyses])) for k in keys}


def aggregate_forehead(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [a["forehead"] for a in analyses if a.get("forehead")]
    if not valid:
        return {"status": "virtual_point", "reason": "hairline segmentation unavailable; forehead_top = glabella + nose length"}
    med = lambda k: [float(np.median([v[k][0] for v in valid])), float(np.median([v[k][1] for v in valid]))]
    return {
        "status": "hairline_segmentation",
        "top_px": med("top_px"), "left_px": med("left_px"), "right_px": med("right_px"),
        "valid_frame_fraction": len(valid) / max(1, len(analyses)),
    }


# -----------------------------------------------------------------------------
# 저장
# -----------------------------------------------------------------------------

def load_reference_10level(csv_path: Path) -> dict[str, Any]:
    """사람 기준 CSV를 읽고 42개 지표와 분위수 경계를 엄격하게 검사한다."""
    source_csv = str(csv_path)
    expected_columns = [
        "feature", "mean", "std", "min", *REFERENCE_PERCENTILES, "max", "count",
    ]
    try:
        if not csv_path.is_file():
            raise FileNotFoundError(f"reference CSV not found: {csv_path}")

        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != expected_columns:
                raise ValueError(
                    "reference CSV columns/order mismatch: "
                    f"expected {expected_columns}, got {reader.fieldnames}"
                )
            rows = list(reader)

        names = [str(row.get("feature", "")).strip() for row in rows]
        if names != list(METRIC_NAMES):
            raise ValueError(
                "reference CSV feature names/order mismatch: "
                f"expected {list(METRIC_NAMES)}, got {names}"
            )

        references: dict[str, dict[str, Any]] = {}
        float_fields = ["mean", "std", "min", *REFERENCE_PERCENTILES, "max"]
        for row in rows:
            feature = str(row["feature"]).strip()
            parsed: dict[str, Any] = {}
            for field in float_fields:
                try:
                    value = float(row[field])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{feature}.{field} is not numeric: {row[field]!r}") from error
                if not math.isfinite(value):
                    raise ValueError(f"{feature}.{field} is not finite: {value!r}")
                parsed[field] = value

            try:
                count_value = float(row["count"])
                count = int(count_value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{feature}.count is not an integer: {row['count']!r}") from error
            if not math.isfinite(count_value) or count_value != count or count <= 0:
                raise ValueError(f"{feature}.count must be a positive integer: {row['count']!r}")
            if parsed["std"] < 0:
                raise ValueError(f"{feature}.std must be non-negative")

            ordered = [parsed["min"], *(parsed[key] for key in REFERENCE_PERCENTILES), parsed["max"]]
            if any(left > right for left, right in zip(ordered, ordered[1:])):
                raise ValueError(f"{feature} percentile boundaries are not ascending")

            parsed["count"] = count
            references[feature] = parsed

        return {
            "status": "available",
            "source_csv": source_csv,
            "metric_count": len(references),
            "references": references,
        }
    except (OSError, ValueError, csv.Error) as error:
        return {
            "status": "unavailable",
            "source_csv": source_csv,
            "metric_count": 0,
            "reason": str(error),
            "references": {},
        }


def build_reference_10level(
    feature_values_ipd: dict[str, float], reference_table: dict[str, Any]
) -> dict[str, Any]:
    """42개 측정값을 p10~p90 경계로 1~10단계 상대 평가한다."""
    if reference_table.get("status") != "available":
        return {
            "status": "unavailable",
            "source_csv": reference_table.get("source_csv"),
            "metric_count": 0,
            "reason": reference_table.get("reason", "reference table unavailable"),
            "metrics": {},
        }

    names = list(feature_values_ipd)
    missing = [name for name in METRIC_NAMES if name not in feature_values_ipd]
    unexpected = [name for name in names if name not in METRIC_NAMES]
    if missing or unexpected:
        return {
            "status": "unavailable",
            "source_csv": reference_table.get("source_csv"),
            "metric_count": 0,
            "reason": (
                "feature_values_ipd names mismatch: "
                f"missing {missing}, unexpected {unexpected}"
            ),
            "metrics": {},
        }

    metrics: dict[str, dict[str, Any]] = {}
    references = reference_table["references"]
    for feature in METRIC_NAMES:
        value = float(feature_values_ipd[feature])
        if not math.isfinite(value):
            return {
                "status": "unavailable",
                "source_csv": reference_table.get("source_csv"),
                "metric_count": 0,
                "reason": f"{feature} measurement is not finite: {value!r}",
                "metrics": {},
            }

        reference = references[feature]
        boundaries = [reference[key] for key in REFERENCE_PERCENTILES]
        level = 1 + sum(value >= boundary for boundary in boundaries)
        if level == 1:
            percentile_band = "value < p10"
        elif level == 10:
            percentile_band = "value >= p90"
        else:
            low = (level - 1) * 10
            high = level * 10
            percentile_band = f"p{low} <= value < p{high}"

        item: dict[str, Any] = {
            "value": value,
            "level": level,
            "percentile_band": percentile_band,
            "reference_mean": reference["mean"],
            "reference_std": reference["std"],
            "reference_count": reference["count"],
        }
        item.update({key: reference[key] for key in REFERENCE_PERCENTILES})
        metrics[feature] = item

    return {
        "status": "available",
        "source_csv": reference_table.get("source_csv"),
        "metric_count": len(metrics),
        "rule": "level 1: value < p10; levels 2-9: pN <= value < p(N+10); level 10: value >= p90",
        "metrics": metrics,
    }

def next_capture_folder(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = directory / stamp
    suffix = 1
    while destination.exists():
        destination = directory / f"{stamp}_{suffix:02d}"
        suffix += 1
    destination.mkdir()
    return destination


def save_capture(directory: Path, coordinates: list[np.ndarray], analyses: list[dict[str, Any]],
                 camera_frame: np.ndarray, screen_image: np.ndarray, measurement: Measurement,
                 reference_table: dict[str, Any], cnn_frames: list[np.ndarray],
                 cnn_face_crops: list[np.ndarray],
                 cnn_face_candidates: list[dict[str, Any]]) -> Path:
    destination = next_capture_folder(directory)
    height, width = camera_frame.shape[:2]
    v41_selected = select_v41_face_crops(cnn_face_candidates)

    quality = {
        "stable": True,
        "sample_count": len(coordinates),
        "average_cv": measurement.score,
        "frontal": True,
        "single_face": True,
        "gaze_verified": True,
        "cnn_multiframe_requested_count": len(cnn_frames),
        "cnn_multiframe_sample_numbers": list(CNN_MULTI_FRAME_SAMPLE_NUMBERS),
        "cnn_face_crop_requested_count": len(cnn_face_crops),
        "cnn_face_crop_margin": CNN_FACE_CROP_MARGIN,
        "cnn_face_v41_candidate_count": len(cnn_face_candidates),
        "cnn_face_v41_selected_count": len(v41_selected),
        "checks": f"yaw<={MAX_YAW_RATIO}, roll<={MAX_ROLL_DEG}deg, pitch_proxy in {PITCH_PROXY_RANGE}, "
                  f"gaze x in {GAZE_X} / y in {GAZE_Y}, eyes open, on every collected frame",
    }
    median_pts = np.median(np.stack(coordinates), axis=0)          # (103,3) 픽셀
    normalized = median_pts.copy()
    normalized[:, 0] /= width
    normalized[:, 1] /= height

    vision = {
        "schema_version": 1,
        "landmark_model": "wflw_mesh.tflite (WFLW 98pt, ours) + face_detection_yunet_2023mar.onnx",
        "point_layout": "0-97 WFLW, 98 forehead_top, 99/100 temple_l/r, 101/102 under_eye_l/r (virtual)",
        "image_width": width, "image_height": height,
        "landmarks": normalized.tolist(),
        "aggregation": "per-point median of pixel coordinates, normalized by image size",
        "quality": quality,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    simple_mean = {k: v for k, v in measurement.mean.items() if not k.startswith("ipd__")}
    simple_std = {k: v for k, v in measurement.std.items() if not k.startswith("ipd__")}
    ipd_mean = {k[5:]: v for k, v in measurement.mean.items() if k.startswith("ipd__")}
    ipd_std = {k[5:]: v for k, v in measurement.std.items() if k.startswith("ipd__")}
    reference_10level = build_reference_10level(ipd_mean, reference_table)

    analysis = dict(analyses[-1])
    analysis.pop("lines", None)
    analysis.update(
        schema_version=2,                                                   # simple/face_fortune.py 의 analysis_result.json 과 같은 구조
        source="face_fortune.py (FaceReading_3 / WFLW)",
        feature_values=simple_mean,                                         # simple 과 같은 36개 (이름·공식 동일)
        feature_std=simple_std,
        eyes=aggregate_simple_eyes(analyses),                               # simple 과 같은 right/left (right = 사진 왼쪽)
        forehead=aggregate_simple_forehead(analyses),                       # simple 과 같은 visible_forehead_* 2개
        quality=quality,
        aggregation="trimmed statistics of per-frame feature ratios",
        image="face_measurement.jpg",
        # ---- FaceReading 체계 (추가 정보) ----
        feature_values_ipd=ipd_mean, feature_std_ipd=ipd_std,
        metric_names_ipd=list(METRIC_NAMES), metric_names_ipd_ko={k: METRIC_KO[k] for k in METRIC_NAMES},
        eyes_wflw=aggregate_eyes(analyses), pose=aggregate_pose(analyses), hairline=aggregate_forehead(analyses),
        reference_10level=reference_10level,
    )
    analysis.pop("eyes_simple", None); analysis.pop("forehead_simple", None)
    handoff = {                                                             # simple 의 handoff.json 과 같은 키 + ipd 추가
        "schema_version": 1,
        "source": "face_fortune.py",
        "feature_values": simple_mean,
        "feature_std": simple_std,
        "eyes": analysis["eyes"],
        "forehead": analysis["forehead"],
        "quality": quality,
        "feature_values_ipd": ipd_mean, "feature_std_ipd": ipd_std, "pose": analysis["pose"], "hairline": analysis["hairline"],
    }

    if not cv2.imwrite(str(destination / "face_measurement.jpg"), camera_frame):
        raise OSError("Could not save face_measurement.jpg")
    if not cv2.imwrite(str(destination / "measurement_screen.jpg"), screen_image):
        raise OSError("Could not save measurement_screen.jpg")
    saved_cnn_frame_names = []
    failed_cnn_frame_names = []
    for index, cnn_frame in enumerate(cnn_frames, start=1):
        frame_name = f"cnn_frame_{index:02d}.jpg"
        if cv2.imwrite(str(destination / frame_name), cnn_frame):
            saved_cnn_frame_names.append(frame_name)
        else:
            failed_cnn_frame_names.append(frame_name)
    saved_cnn_face_names = []
    failed_cnn_face_names = []
    for index, face_crop in enumerate(cnn_face_crops, start=1):
        face_name = f"cnn_face_{index:02d}.jpg"
        if cv2.imwrite(str(destination / face_name), face_crop):
            saved_cnn_face_names.append(face_name)
        else:
            failed_cnn_face_names.append(face_name)

    saved_v41_face_names = []
    failed_v41_face_names = []
    v41_selection = []
    for index, item in enumerate(v41_selected, start=1):
        face_name = f"cnn_face_v41_{index:02d}.jpg"
        if cv2.imwrite(str(destination / face_name), item["crop"]):
            saved_v41_face_names.append(face_name)
        else:
            failed_v41_face_names.append(face_name)
        v41_selection.append({
            "image": face_name,
            "sample_number": int(item["sample_number"]),
            "sharpness_laplacian_variance": float(item["sharpness"]),
        })

    cnn_manifest = {
        "schema_version": 1,
        "purpose": "multi-frame CNN diagnostic inputs for V3, V4 and V4.1",
        "selection": "V3/V4 use accepted frame numbers 1, 8, 15, 22, 30; V4.1 picks the sharpest face crop inside each of five temporal bins",
        "requested_count": len(cnn_frames),
        "saved_count": len(saved_cnn_frame_names),
        "images": saved_cnn_frame_names,
        "failed_images": failed_cnn_frame_names,
        "face_crop_margin": CNN_FACE_CROP_MARGIN,
        "face_crop_size": [IMG, IMG],
        "face_crop_saved_count": len(saved_cnn_face_names),
        "face_crop_images": saved_cnn_face_names,
        "failed_face_crop_images": failed_cnn_face_names,
        "v41_temporal_bins": [list(item) for item in CNN_FACE_V41_BIN_RANGES],
        "v41_candidate_count": len(cnn_face_candidates),
        "v41_saved_count": len(saved_v41_face_names),
        "v41_images": saved_v41_face_names,
        "v41_failed_images": failed_v41_face_names,
        "v41_selection": v41_selection,
        "v41_sharpness_method": "variance of Laplacian on the 160x160 WFLW face crop",
    }
    reference_payload = {
        "schema_version": 1,
        "source": "face_fortune.py (FaceReading_3 / WFLW)",
        "capture_id": destination.name,
        "captured_at_utc": vision["captured_at_utc"],
        **reference_10level,
    }
    for name, data in (
        ("vision_landmarks.json", vision),
        ("analysis_result.json", analysis),
        ("handoff.json", handoff),
        ("face_reference_10level.json", reference_payload),
        ("cnn_frames_manifest.json", cnn_manifest),
    ):
        (destination / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return destination


# -----------------------------------------------------------------------------
# Animal + Pokemon result saving
# -----------------------------------------------------------------------------

def save_animal_pokemon_result(
    saved_path: Path,
    result_data: dict[str, Any],
) -> Path:
    pokemon_results = result_data.get("pokemon_results", [])
    human_style = result_data.get("human_style", {})
    z_scores = result_data.get("z_scores", {})
    quality = result_data.get("quality", {})
    score_diagnostics = result_data.get("score_diagnostics", {})

    payload = {
        "schema_version": 2,
        "source": "face_fortune.py (FaceReading_3 / WFLW)",
        "capture_id": saved_path.name,
        "animal_score_semantics": (
            "independent 0-100 MATCH score; not probability; TOP3 does not sum to 100"
        ),
        "animal_pokemon_top3": [
            {
                "rank": index,
                "animal": str(item["animal"]),
                # Compatibility: animal_percent now stores the independent
                # MATCH score rather than a normalized TOP3 share.
                "animal_percent": float(item["animal_percent"]),
                "animal_match_score": float(
                    item.get("animal_match_score", item["animal_percent"])
                ),
                "pokemon": str(item["pokemon"]),
            }
            for index, item in enumerate(
                pokemon_results,
                start=1,
            )
        ],
        "human_style": {
            str(key): float(value)
            for key, value in human_style.items()
        },
        "geometry_z_scores": {
            str(key): float(value)
            for key, value in z_scores.items()
        },
        "animal_score_diagnostics": score_diagnostics,
        "quality": quality,
    }

    result_path = saved_path / "animal_pokemon_result.json"
    result_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return result_path


def analyze_and_save_animal_pokemon(
    saved_path: Path,
) -> dict[str, Any]:
    result_data = analyze_capture_folder(saved_path)

    print()
    print("=== ANIMAL + POKEMON RESULT (V4.1 BALANCED FINAL) ===")

    for index, item in enumerate(
        result_data["pokemon_results"],
        start=1,
    ):
        print(
            f"{index}. "
            f"{item['animal']:10s} "
            f"MATCH {item['animal_percent']:.1f} "
            f"-> "
            f"{item['pokemon']}"
        )

    production = (
        result_data.get("score_diagnostics", {})
        .get("production_selection", {})
    )
    v41 = (
        result_data.get("score_diagnostics", {})
        .get("wflw_facecrop_sharp_v41", {})
    )
    aggregation = v41.get("cnn_aggregation", {})

    if production.get("status") == "v4.1_active":
        print(
            "Method: sharpness-selected face crops x5 "
            "| CNN 60% + WFLW geometry 40% "
            "| mean cohesion:",
            f"{aggregation.get('mean_frame_to_aggregate_cosine', 0.0):.4f}",
        )
    else:
        print(
            "WARNING: V4.1 unavailable; legacy fallback used:",
            production.get("reason", "unknown reason"),
        )

    result_path = save_animal_pokemon_result(
        saved_path,
        result_data,
    )
    print("Result saved:", result_path)
    return result_data


# -----------------------------------------------------------------------------
# 카메라
# -----------------------------------------------------------------------------

def open_camera(preferred: int) -> cv2.VideoCapture:
    order = (preferred,) + tuple(i for i in CAMERA_FALLBACK if i != preferred)
    for index in order:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release(); continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = cap.read()
        if not ok:
            cap.release(); continue
        if index != preferred:
            print(f"Camera {preferred} unavailable; using /dev/video{index}")
        return cap
    raise RuntimeError(f"Cannot open any camera (tried {preferred} then {CAMERA_FALLBACK}). Check: v4l2-ctl --list-devices")


# -----------------------------------------------------------------------------
# 한글 텍스트 (PIL)  -  보드에 한글 폰트가 있으면 쓰고, 없으면 영문 라벨로 대체
# -----------------------------------------------------------------------------

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except ImportError:                                                     # pillow 없음 → 영문 fallback
    _PIL_OK = False

KOREAN_FONT_CANDIDATES = [
    str(Path(__file__).resolve().parent / "NanumGothic.ttf"),           # 같은 폴더에 두면 최우선
    str(Path(__file__).resolve().parent / "NanumGothicBold.ttf"),
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "C:/Windows/Fonts/malgunbd.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
KOREAN_FONT = next((p for p in KOREAN_FONT_CANDIDATES if os.path.exists(p)), None) if _PIL_OK else None
_FONT_CACHE: dict[int, Any] = {}


def _kfont(size: int):
    font = _FONT_CACHE.get(size)
    if font is None:
        font = ImageFont.truetype(KOREAN_FONT, size)
        _FONT_CACHE[size] = font
    return font


def draw_ktext(img, text, x, y, size=20, color=(235, 235, 240), fallback=None, center=False):
    """한글 텍스트를 img 에 그린다. (x, y) 는 글자 상자의 왼쪽 위 (center=True 면 x 가 가운데).
    한글 폰트가 없으면 fallback(영문) 을 Hershey 폰트로 그린다. 글자 영역(ROI)만 PIL 로 변환해 느리지 않다."""
    if KOREAN_FONT is None:
        label = fallback or text
        scale = size / 36.0
        if center:
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            x = int(x - tw / 2)
        draw_text(img, label, int(x), int(y + size * 0.85), scale, color, 2 if size >= 20 else 1)
        return
    font = _kfont(size)
    left, top, right, bottom = font.getbbox(text)
    tw, th = right - left, bottom - top
    if center:
        x = x - tw / 2
    x, y = int(x), int(y)
    H, W = img.shape[:2]
    x0, y0 = max(0, x - 2), max(0, y - 2)
    x1, y1 = min(W, x + tw + 4), min(H, y + th + top + 6)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text((x - x0, y - y0), text, font=font, fill=(int(color[2]), int(color[1]), int(color[0])))
    img[y0:y1, x0:x1] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


# -----------------------------------------------------------------------------
# 화면  -  모던 다크 테마 (둥근 모서리 · 평면 · 단일 강조색)
# -----------------------------------------------------------------------------

# 색 (BGR)
C_BG       = (23, 20, 18)        # 바탕      #12141 7
C_SURFACE  = (36, 31, 28)        # 카드      #1C1F24
C_SURFACE2 = (49, 42, 38)        # 카드 안쪽 #262A31
C_BORDER   = (62, 54, 48)        # 테두리    #30363E
C_TEXT     = (247, 244, 242)     # 본문      #F2F4F7
C_MUTED    = (161, 147, 139)     # 보조 글씨 #8B93A1
C_DIM      = (105, 96, 90)       # 흐린 글씨 #5A6069
C_MINT     = (128, 222, 74)      # 강조 1    #4ADE80
C_BLUE     = (250, 165, 96)      # 강조 2    #60A5FA
C_AMBER    = (36, 191, 251)      # 경고      #FBBF24
C_RED      = (113, 113, 248)     # 오류      #F87171

WINDOW_NAME = "Face Fortune"
PANEL_W = 520

MODE_BUTTONS = [
    {"id": "gwansang",  "label": "관상",   "fallback": "GWANSANG",  "accent": C_MINT, "key": "1"},
    {"id": "lookalike", "label": "닮은꼴", "fallback": "LOOKALIKE", "accent": C_BLUE, "key": "2"},
]
SEG_X, SEG_Y, SEG_H, SEG_PAD = 20, 84, 52, 4         # 세그먼트 컨트롤 위치·높이·안쪽 여백

ui_state: dict[str, Any] = {"mode": None, "hover": None, "pressed": None, "panel_x": 0}


def draw_text(img, text, x, y, scale=0.44, color=C_TEXT, thickness=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def rounded_rect(img, x1, y1, x2, y2, r, color, thickness=-1):
    """둥근 사각형. thickness=-1 이면 채움."""
    r = int(max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2)))
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
        return
    t = thickness
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, t, cv2.LINE_AA)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, t, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, t, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, t, cv2.LINE_AA)
    for (cx, cy, a) in ((x1 + r, y1 + r, 180), (x2 - r, y1 + r, 270), (x2 - r, y2 - r, 0), (x1 + r, y2 - r, 90)):
        cv2.ellipse(img, (cx, cy), (r, r), a, 0, 90, color, t, cv2.LINE_AA)


def blend_rect(img, x1, y1, x2, y2, color, alpha):
    """반투명 사각형"""
    roi = img[y1:y2, x1:x2]
    layer = np.empty_like(roi); layer[:] = color
    cv2.addWeighted(layer, alpha, roi, 1 - alpha, 0, roi)


def status_color(status: str):
    s = status.upper()
    if "SAVED" in s or "READY" in s: return C_MINT
    if "COLLECTING" in s or "AUTO SAVING" in s: return C_BLUE
    if any(w in s for w in ("BLINK", "LOOK", "MOVE", "MOVED", "KEEP", "HOLD")): return C_AMBER
    if any(w in s for w in ("UNSTABLE", "REQUIRED", "FAILED")): return C_RED
    return C_MUTED


def status_korean(status: str) -> str:
    s = status.upper()
    table = (
        ("MOVED - HOLD STILL", "얼굴이 움직였어요 · 정면을 보고 잠시 그대로 유지해 주세요"),
        ("ONE FACE REQUIRED", "한 사람만 화면에 들어와 주세요"),
        ("KEEP FACE INSIDE", "얼굴 전체가 화면 안에 들어오도록 맞춰 주세요"),
        ("MOVE CLOSER", "조금 더 가까이 와 주세요"),
        ("LOOK STRAIGHT", "정면을 봐 주세요"),
        ("BLINK", "눈을 뜨고 잠시 그대로 있어 주세요"),
        ("LOOK AT THE CAMERA", "카메라 렌즈를 바라봐 주세요"),
        ("AUTO-SAVED", "촬영 완료 · 결과를 준비하고 있어요"),
        ("AUTO SAVING", "측정 완료 · 저장 중"),
        ("UNSTABLE", "움직임이 컸어요 · 정면을 보고 다시 유지해 주세요"),
        ("SAVE FAILED", "저장에 실패했어요"),
        ("COLLECTING", "좋아요 · 정면을 보고 움직이지 말고 유지해 주세요"),
    )
    return next((ko for token, ko in table if token in s), status)


def draw_corner_brackets(img, x1, y1, x2, y2, color):
    arm, t = 22, 2
    for (ax, ay, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (ax, ay), (ax + dx * arm, ay), color, t, cv2.LINE_AA)
        cv2.line(img, (ax, ay), (ax, ay + dy * arm), color, t, cv2.LINE_AA)


def draw_progress(img, x, y, w, h, value, total, color=C_MINT):
    """얇은 둥근 진행 바"""
    rounded_rect(img, x, y, x + w, y + h, h // 2, C_SURFACE2)
    fill = int(w * min(1.0, value / max(total, 1)))
    if fill > h:
        rounded_rect(img, x, y, x + fill, y + h, h // 2, color)


def draw_landmarks(canvas, pts, status, hairline):
    ipt = lambda i: (int(pts[i, 0]), int(pts[i, 1]))
    for name, a, b in MEASURE_LINES:                                   # 측정선 (헤어라인이 있으면 얼굴 길이는 헤어라인 상단부터)
        pa = tuple(map(int, hairline["top"])) if (name == "face_len" and hairline is not None) else ipt(a)
        cv2.line(canvas, pa, ipt(b), (90, 80, 70), 1, cv2.LINE_AA)
    for chain, closed in ((CONTOUR, False), (MOUTH_OUTER, True), (BROWS["left"], False), (BROWS["right"], False)):
        cv2.polylines(canvas, [np.array([ipt(i) for i in chain])], closed, C_MINT, 1, cv2.LINE_AA)
    for e in EYES.values():
        cv2.polylines(canvas, [np.array([ipt(i) for i in e["ring"]])], True, C_BLUE, 1, cv2.LINE_AA)
        cv2.circle(canvas, ipt(e["pupil"]), 2, C_TEXT, -1, cv2.LINE_AA)
    for i in range(98):
        cv2.circle(canvas, ipt(i), 1, (200, 200, 200), -1)
    virtual_ids = range(101, N_POINTS) if hairline is not None else range(98, N_POINTS)
    for i in virtual_ids:
        cv2.circle(canvas, ipt(i), 4, C_AMBER, 1, cv2.LINE_AA)
    if hairline is not None:
        cv2.line(canvas, tuple(map(int, hairline["left"])), tuple(map(int, hairline["right"])), C_MINT, 2, cv2.LINE_AA)
        cv2.circle(canvas, tuple(map(int, hairline["top"])), 5, C_MINT, -1, cv2.LINE_AA)
    x1, y1, x2, y2 = face_bbox(pts)
    h, w = canvas.shape[:2]
    draw_corner_brackets(canvas, max(6, x1 - 16), max(6, y1 - 16), min(w - 7, x2 + 16), min(h - 7, y2 + 16), status_color(status))


# ---- 오른쪽 패널 버튼 ---------------------------------------------------------

def button_rect(index: int):
    """index 번째 버튼의 패널 기준 사각형 (x1, y1, x2, y2)  -  세그먼트 컨트롤 안의 칸"""
    n = len(MODE_BUTTONS)
    inner_w = PANEL_W - SEG_X * 2 - SEG_PAD * 2
    bw = inner_w // n
    x1 = SEG_X + SEG_PAD + index * bw
    return x1, SEG_Y + SEG_PAD, x1 + bw, SEG_Y + SEG_H - SEG_PAD


def button_at(px: int, py: int):
    for i, b in enumerate(MODE_BUTTONS):
        x1, y1, x2, y2 = button_rect(i)
        if x1 <= px <= x2 and y1 <= py <= y2:
            return b["id"]
    return None


def select_mode(mode_id):
    """버튼 선택. 이미 선택된 버튼을 다시 누르면 해제."""
    ui_state["mode"] = None if ui_state["mode"] == mode_id else mode_id


def on_mouse(event, x, y, flags, param):
    hit = button_at(x - ui_state["panel_x"], y)
    if event == cv2.EVENT_MOUSEMOVE:
        ui_state["hover"] = hit
    elif event == cv2.EVENT_LBUTTONDOWN:
        ui_state["pressed"] = hit
    elif event == cv2.EVENT_LBUTTONUP:
        if hit is not None and hit == ui_state["pressed"]:
            select_mode(hit)
        ui_state["pressed"] = None


def draw_buttons(panel):
    """알약 모양 세그먼트 컨트롤: 선택된 칸만 강조색으로 채움"""
    rounded_rect(panel, SEG_X, SEG_Y, PANEL_W - SEG_X, SEG_Y + SEG_H, SEG_H // 2, C_SURFACE)
    for i, b in enumerate(MODE_BUTTONS):
        x1, y1, x2, y2 = button_rect(i)
        active = ui_state["mode"] == b["id"]
        hover = ui_state["hover"] == b["id"]
        pressed = ui_state["pressed"] == b["id"]
        r = (y2 - y1) // 2
        if active or pressed:
            fill = tuple(int(c * (0.78 if pressed else 1.0)) for c in b["accent"])
            rounded_rect(panel, x1, y1, x2, y2, r, fill)
            text_color = C_BG
        elif hover:
            rounded_rect(panel, x1, y1, x2, y2, r, C_SURFACE2)
            text_color = C_TEXT
        else:
            text_color = C_MUTED
        draw_ktext(panel, b["label"], (x1 + x2) // 2, y1 + 10, 22, text_color, b["fallback"], center=True)


def draw_mode_panel(panel, y, h):
    """선택된 버튼의 정보 영역 (내용은 추후 연결)"""
    mode = next((b for b in MODE_BUTTONS if b["id"] == ui_state["mode"]), None)
    x1, x2 = SEG_X, PANEL_W - SEG_X
    rounded_rect(panel, x1, y, x2, y + h, 16, C_SURFACE)
    if mode is None:
        cy = y + h // 2
        cv2.circle(panel, (PANEL_W // 2, cy - 44), 22, C_SURFACE2, -1, cv2.LINE_AA)
        draw_text(panel, "?", PANEL_W // 2 - 8, cy - 35, 0.75, C_DIM, 2)
        draw_ktext(panel, "관상 또는 닮은꼴을 선택하세요", PANEL_W // 2, cy, 18, C_MUTED, "Select a mode above", center=True)
        draw_ktext(panel, "결과가 이 영역에 표시됩니다", PANEL_W // 2, cy + 30, 14, C_DIM, "Results will appear here", center=True)
        return
    px = x1 + 24
    cv2.circle(panel, (px + 5, y + 36), 5, mode["accent"], -1, cv2.LINE_AA)
    draw_ktext(panel, mode["label"], px + 22, y + 20, 26, C_TEXT, mode["fallback"])
    draw_text(panel, mode["fallback"], px + 22, y + 72, 0.34, mode["accent"], 1)
    cv2.line(panel, (px, y + 90), (x2 - 24, y + 90), C_BORDER, 1, cv2.LINE_AA)
    rounded_rect(panel, px, y + 108, x2 - 24, y + 156, 10, C_SURFACE2)
    draw_ktext(panel, "정보 영역 · 추후 연결 예정", px + 16, y + 122, 15, C_MUTED, "Info area - coming soon")


# ---- 전체 화면 ------------------------------------------------------------------

def draw(frame, pts, analysis, status, count, hairline=None, saved_path=None, saved_age=None, save_error=None):
    height, width = frame.shape[:2]
    display_height = max(height, 760)
    canvas = np.zeros((display_height, width, 3), np.uint8); canvas[:] = C_BG
    canvas[:height] = frame

    if pts is not None:
        draw_landmarks(canvas, pts, status, hairline)

    # 영상 위 상단: 반투명 띠 + 제목
    blend_rect(canvas, 0, 0, width, 48, C_BG, 0.55)
    draw_text(canvas, "FACE FORTUNE", 18, 31, 0.62, C_TEXT, 2)
    draw_text(canvas, "WFLW 98pt", width - 96, 30, 0.36, C_MUTED, 1)

    # 영상 아래: 상태 카드
    sc = status_color(status)
    cy0 = height + 16
    rounded_rect(canvas, 16, cy0, width - 16, display_height - 16, 16, C_SURFACE)
    cv2.circle(canvas, (44, cy0 + 36), 6, sc, -1, cv2.LINE_AA)
    draw_ktext(canvas, status_korean(status), 62, cy0 + 22, 22, C_TEXT, status)
    draw_text(canvas, status, 62, cy0 + 70, 0.34, C_MUTED, 1)
    draw_text(canvas, f"{count:02d} / {TARGET_FRAMES:02d}", width - 92, cy0 + 112, 0.42, C_TEXT, 1)
    draw_ktext(canvas, "스캔 진행", 40, cy0 + 96, 14, C_DIM, "scan")
    draw_progress(canvas, 40, cy0 + 118, width - 160, 8, count, TARGET_FRAMES, sc if count else C_MINT)
    draw_ktext(canvas, "움직이지 말고 카메라 렌즈를 봐 주세요", 40, cy0 + 146, 14, C_DIM, "Hold still. Keep eyes on the lens.")

    # 오른쪽 패널
    pw = PANEL_W
    ui_state["panel_x"] = width
    panel = np.zeros((display_height, pw, 3), np.uint8); panel[:] = C_BG
    cv2.line(panel, (0, 0), (0, display_height), C_BORDER, 1)
    draw_text(panel, "FACE READING", SEG_X, 31, 0.62, C_TEXT, 2)
    draw_ktext(panel, "관상 · 닮은꼴", SEG_X + 196, 15, 16, C_MUTED, "GWANSANG / LOOKALIKE")
    draw_text(panel, "AUTO SAVE", pw - 96, 30, 0.34, C_MINT, 1)

    draw_buttons(panel)
    fy = display_height - 40
    content_y = SEG_Y + SEG_H + 16
    draw_mode_panel(panel, content_y, fy - 16 - content_y)

    draw_text(panel, "R  new scan      S  save      1 / 2  mode      Q  quit", SEG_X, fy + 26, 0.34, C_DIM, 1)

    merged = np.hstack((canvas, panel))
    # 저장 알림 (토스트)
    if saved_path is not None and (saved_age is None or saved_age < 4.5):
        bw, bh = 460, 72; x1 = (merged.shape[1] - bw) // 2; y1 = 60
        rounded_rect(merged, x1, y1, x1 + bw, y1 + bh, 14, C_SURFACE)
        rounded_rect(merged, x1, y1, x1 + bw, y1 + bh, 14, C_MINT, 1)
        cv2.circle(merged, (x1 + 30, y1 + 36), 9, C_MINT, -1, cv2.LINE_AA)
        draw_ktext(merged, "저장 완료", x1 + 52, y1 + 14, 20, C_TEXT, "SAVED")
        draw_text(merged, saved_path.name, x1 + 52, y1 + 58, 0.38, C_MUTED, 1)
    elif save_error:
        bw, bh = 460, 72; x1 = (merged.shape[1] - bw) // 2; y1 = 60
        rounded_rect(merged, x1, y1, x1 + bw, y1 + bh, 14, C_SURFACE)
        rounded_rect(merged, x1, y1, x1 + bw, y1 + bh, 14, C_RED, 1)
        cv2.circle(merged, (x1 + 30, y1 + 36), 9, C_RED, -1, cv2.LINE_AA)
        draw_ktext(merged, "저장 실패", x1 + 52, y1 + 14, 20, C_TEXT, "SAVE FAILED")
        draw_text(merged, str(save_error)[:58], x1 + 52, y1 + 58, 0.36, C_MUTED, 1)
    return merged


# -----------------------------------------------------------------------------
# Web live analysis preview
# -----------------------------------------------------------------------------

def write_live_analysis_preview(
    frame_bgr: np.ndarray,
    status: str,
    count: int,
    frame_number: int,
) -> None:
    """웹 분석 화면에서 읽을 현재 프레임/상태를 원자적으로 갱신한다.

    내부 분석은 랜드마크/측정선을 계속 사용하지만,
    웹 키오스크에는 사용자가 보는 일반 카메라 원본 프레임만 저장한다.
    """
    LIVE_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    preview = frame_bgr.copy()

    image_tmp = LIVE_PREVIEW_DIR / ".live_analysis.tmp.jpg"
    state_tmp = LIVE_PREVIEW_DIR / ".live_analysis.tmp.json"

    ok = cv2.imwrite(
        str(image_tmp),
        preview,
        [int(cv2.IMWRITE_JPEG_QUALITY), LIVE_PREVIEW_JPEG_QUALITY],
    )
    if ok:
        os.replace(image_tmp, LIVE_PREVIEW_IMAGE)

    payload = {
        "status": status,
        "status_ko": status_korean(status),
        "collected": int(count),
        "target": int(TARGET_FRAMES),
        "frame_number": int(frame_number),
        "updated_at": time.time(),
    }
    state_tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(state_tmp, LIVE_PREVIEW_STATE)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--headless", action="store_true", help="창 없이 (SSH 테스트용)")
    parser.add_argument("--max-frames", type=int, default=0, help="이 프레임 수 뒤 종료 (0=무제한)")
    parser.add_argument("--output-dir", type=Path, default=CAPTURE_DIR)
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=REFERENCE_CSV,
        help="42개 사람 기준 p10~p90 CSV",
    )
    parser.add_argument("--no-hairline", action="store_true", help="헤어라인 분할 끄기 (가상 이마 점만 사용)")
    parser.add_argument("--no-eye-refine", action="store_true", help="2단계 눈 정밀 모델 끄기 (1단계 눈 점 그대로)")
    args = parser.parse_args()

    LIVE_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for stale_path in (LIVE_PREVIEW_IMAGE, LIVE_PREVIEW_STATE):
        try:
            stale_path.unlink(missing_ok=True)
        except OSError:
            pass

    reference_table = load_reference_10level(args.reference_csv)
    if reference_table["status"] == "available":
        print(
            f"Reference 10-level loaded: {reference_table['source_csv']} "
            f"({reference_table['metric_count']} metrics)"
        )
    else:
        print(
            "Reference 10-level unavailable; capture will continue: "
            f"{reference_table['reason']}"
        )

    landmarker = Landmarker(eye_refine=not args.no_eye_refine)
    hair = Hairline(enabled=not args.no_hairline)
    cap = open_camera(args.camera)

    measurement = Measurement()
    coordinates: list[np.ndarray] = []
    analyses: list[dict[str, Any]] = []
    cnn_frames: list[np.ndarray] = []
    cnn_face_crops: list[np.ndarray] = []
    cnn_face_candidates: list[dict[str, Any]] = []
    save_state: dict[str, Any] = {"path": None, "saved_at": None, "error": None}
    ready_screen = ready_frame = None
    last_hairline = None
    invalid_streak = frames = 0
    movement_warning_until = 0
    last_live_preview_at = 0.0

    def reset_scan() -> None:
        nonlocal ready_screen, ready_frame, invalid_streak, last_hairline, movement_warning_until
        measurement.reset(); coordinates.clear(); analyses.clear(); cnn_frames.clear(); cnn_face_crops.clear(); cnn_face_candidates.clear()
        save_state.update(path=None, saved_at=None, error=None)
        ready_screen = ready_frame = None; last_hairline = None; invalid_streak = 0
        movement_warning_until = 0

    def do_save(frame_for_save, screen_for_save, tag):
        try:
            p = save_capture(
                args.output_dir, coordinates, analyses, frame_for_save, screen_for_save,
                measurement, reference_table, cnn_frames, cnn_face_crops, cnn_face_candidates,
            )
            save_state.update(path=p, saved_at=time.monotonic(), error=None)
            v = measurement.mean
            print(f"{tag}: {p}")
            print("  face_hw=%.3f cheek/jaw=%.3f jaw_angle=%.1f nose_hw=%.3f mouth_w=%.3f  (cv=%.4f)"
                  % (v["face_hw_ratio"], v["cheek_to_jaw_width_ratio"], v["jaw_angle_deg"], v["nose_hw_ratio"], v["mouth_width_ratio"], measurement.score))

            try:
                analyze_and_save_animal_pokemon(p)
            except Exception as error:
                print("Animal/Pokemon analysis failed:", error)
        except OSError as error:
            save_state.update(error=str(error)); print("Save failed:", error)

    print(f"Collecting {TARGET_FRAMES} frames per scan. Keys: R new / S save / 1 gwansang / 2 lookalike / Q quit")
    if KOREAN_FONT is None:
        print("Korean font not found; using English button labels (put NanumGothic.ttf next to this file to fix)")
    if not args.headless:
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed"); break
            frame = cv2.flip(frame, 1)                       # 거울처럼
            frames += 1
            height, width = frame.shape[:2]

            pts = landmarker(frame)
            analysis = None; valid = False; blink = False; completed = False
            status = "ONE FACE REQUIRED"

            if pts is not None:
                x1, y1, x2, y2 = face_bbox(pts)
                fits = x1 > width * 0.02 and y1 > height * 0.02 and x2 < width * 0.98 and y2 < height * 0.98
                large = (pts[P["cheek_r"], 0] - pts[P["cheek_l"], 0]) >= width * MIN_FACE_WIDTH_RATIO
                eyes = {side: eye_metrics(pts, side) for side in EYES}
                blink = not eyes_open(eyes)
                pose = pose_of(pts)
                frontal = is_frontal(pose)
                gaze_ok = True if blink else gaze_centered(eyes)
                valid = fits and large and frontal and gaze_ok and not blink

                if not fits:        status = "KEEP FACE INSIDE FRAME"
                elif not large:     status = "MOVE CLOSER"
                elif not frontal:   status = "LOOK STRAIGHT AT CAMERA"
                elif blink:         status = "BLINK - HOLD ON"
                elif not gaze_ok:   status = "LOOK AT THE CAMERA LENS"
                else:               status = "COLLECTING - HOLD STILL"

                if valid:
                    if hair.ok and (last_hairline is None or frames % FOREHEAD_EVERY == 0):
                        last_hairline = hair(frame, pts)
                    analysis = analyze_frame(pts, last_hairline, hair.last_mask if last_hairline is not None else None)
                else:
                    analysis = analyze_frame(pts, None)

            if blink:
                pass
            elif not valid:
                invalid_streak += 1
                if invalid_streak >= RESET_AFTER_INVALID:
                    reset_scan()
            else:
                invalid_streak = 0
                before = len(measurement.samples)
                merged = dict(analysis["feature_values"]); merged.update({"ipd__" + k: v for k, v in analysis["feature_values_ipd"].items()})
                completed = measurement.observe(merged, face_center(pts), width)
                after = len(measurement.samples)
                if after > before:
                    coordinates.append(pts.copy()); analyses.append(analysis)
                    face_crop = cnn_face_crop(frame, pts)
                    cnn_face_candidates.append({
                        "sample_number": int(after),
                        "crop": face_crop.copy(),
                        "sharpness": cnn_face_sharpness(face_crop),
                    })
                    if after in CNN_MULTI_FRAME_SAMPLE_NUMBERS:
                        cnn_frames.append(frame.copy())
                        cnn_face_crops.append(face_crop.copy())
                if after < before:                                     # 중심이 튀어 reset 된 경우
                    movement_warning_until = frames + 20
                    status = "MOVED - HOLD STILL AND LOOK STRAIGHT"
                    del coordinates[:]; del analyses[:]; del cnn_frames[:]; del cnn_face_crops[:]; del cnn_face_candidates[:]
                    if after == 1:
                        coordinates.append(pts.copy()); analyses.append(analysis)
                        face_crop = cnn_face_crop(frame, pts)
                        cnn_face_candidates.append({
                            "sample_number": int(after),
                            "crop": face_crop.copy(),
                            "sharpness": cnn_face_sharpness(face_crop),
                        })
                        if after in CNN_MULTI_FRAME_SAMPLE_NUMBERS:
                            cnn_frames.append(frame.copy())
                            cnn_face_crops.append(face_crop.copy())
                    ready_screen = ready_frame = None
                    save_state.update(path=None, saved_at=None, error=None)
                if after == TARGET_FRAMES:
                    status = "READY - AUTO SAVING" if measurement.stable else "UNSTABLE - R TO RETRY"

            # 얼굴 중심이 갑자기 움직여 샘플이 초기화된 경우,
            # 사용자가 왜 다시 수집되는지 알 수 있도록 잠시 경고를 유지한다.
            if (
                frames < movement_warning_until
                and valid
                and len(measurement.samples) < TARGET_FRAMES
            ):
                status = "MOVED - HOLD STILL AND LOOK STRAIGHT"

            saved_age = None
            if save_state["saved_at"] is not None:
                saved_age = time.monotonic() - save_state["saved_at"]
                if save_state["path"] is not None:
                    status = "AUTO-SAVED - PRESS R FOR NEW SCAN"

            display = draw(frame, pts, analysis, status, len(measurement.samples), last_hairline if valid else None,
                           save_state["path"], saved_age, save_state["error"])

            now_live = time.monotonic()
            if now_live - last_live_preview_at >= LIVE_PREVIEW_INTERVAL_SEC:
                try:
                    write_live_analysis_preview(
                        frame_bgr=frame,
                        status=status,
                        count=len(measurement.samples),
                        frame_number=frames,
                    )
                    last_live_preview_at = now_live
                except Exception as error:
                    # 웹 미리보기 실패가 실제 얼굴 분석을 중단시키면 안 된다.
                    print("Live preview update failed:", error)

            if completed or (len(measurement.samples) == TARGET_FRAMES and measurement.stable):
                ready_screen, ready_frame = display.copy(), frame.copy()

            if (len(measurement.samples) == TARGET_FRAMES and measurement.stable and ready_screen is not None
                    and save_state["path"] is None and save_state["error"] is None):
                do_save(ready_frame, ready_screen, "Auto-saved")
                display = draw(frame, pts, analysis, "AUTO-SAVED - PRESS R FOR NEW SCAN" if save_state["path"] else "SAVE FAILED",
                               len(measurement.samples), None, save_state["path"], 0.0, save_state["error"])

            if not args.headless:
                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"): break
                if key == ord("r"): reset_scan()
                for b in MODE_BUTTONS:
                    if key == ord(b["key"]):
                        select_mode(b["id"])
                if key == ord("s"):
                    if measurement.stable and ready_frame is not None:
                        do_save(ready_frame, ready_screen, "Saved again")
                    else:
                        print("Measurement not ready")

            if args.max_frames and frames >= args.max_frames:
                break
        print(f"Processed {frames} frames; collected {len(measurement.samples)}")
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
