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
    eye_refine.tflite                   (선택) 2단계 눈 정밀 모델 (EyeRefine_Train.ipynb). 눈 8점+눈썹 9점+눈동자 1점을 다시 찍는다.
                                        1단계 모델이 눈·눈썹을 평균 모양으로 뭉개는 문제(눈꼬리 각도 측정 불가) 해결용. 없으면 1단계 점 그대로

실행:  python face_fortune.py            (카메라 기본 1번, 안 열리면 0·2번 자동 시도)
키  :  R 새로 스캔 / S 다시 저장 / V 관상 풀이 다시 보기 / Q 종료 / 1 관상 / 2 닮은꼴
마우스:  오른쪽 패널의 [관상] 버튼 → 실시간 라벨 + 풀이 다시 보기,  [닮은꼴] 은 아직 내용 없음

관상 풀이 (이 파일에 들어 있음):  스캔이 끝나 저장되면 '관상 풀이' 창이 따로 뜬다.  창에는 눈 유형·부위 칩, 한눈에 보기, 얼굴형(오행)(+종합 등급), 운세 7가지가 나온다.
    전체 내용 (운세·십이궁·측정값 42개 등)은 캡처 폴더의 reading.txt.   ↑↓ / 휠 스크롤, Q 닫기.   저장된 캡처 다시 보기:  python face_fortune_labels2.py --view captures/폴더
    창 대신 터미널로 보려면 --terminal.  창 한글: NanumGothic.ttf + pillow 필요 (없으면 자동으로 터미널 출력)
5단계 기준:  눈꼬리 각도 (왼쪽·오른쪽 눈 중 큰 값)  <5.0 / 5.0~7.3 / 7.4~9.6 / 9.7~12.0 / >12.0 °   (평균 8.5°)
            두 눈 사이  (두 눈 사이 ÷ 눈 가로 길이)  <1.15 / 1.15~1.30 / 1.30~1.45 / 1.45~1.60 / >1.60   (평균 1.38)
            눈 열림·입은 face_reference*.csv 가 있으면 그 20%씩, 없으면 문서의 논문 평균±편차
그 밖의 라벨: 이마 폭·광대 폭·좌우 균형·눈 크기·눈썹 길이(형제궁)·눈썹과 눈 사이(전택궁)·미간(명궁) (기준표 백분위, 없으면 임시 평균),
            운세 7가지(재물·애정·건강·직업·인간관계·가정·말년)와 부족한 기운(오행)·지닐 것·피할 것,
            입술 두께(÷입 너비)·윗/아랫입술 균형·입꼬리(입이 벌어지면 ? 표시), 얼굴 가로세로 비율(세로÷가로, 평균 1.38 기준 임시), 눈썹 각도(<0 내려감 / 0~5 일자 / 5~12 평균 / 12~ 치켜), 코 길이(코 길이 ÷ 광대 폭, 임시 기준), 인중, 입 크기, 눈 열림.  코 너비는 뺐다.
AI:         이 버전은 Qwen(AI)을 쓰지 않는다 (Ollama 불필요, 네트워크 요청 없음).
규칙집:      gwansang_rules.json 을 이 파일 옆에 두면 오행 얼굴형·삼정이 나오고, face_reference*.csv 도 있으면 십이궁·운세 5가지·특징·종합 등급까지 나온다.
쌍꺼풀 (선택): eyelid.tflite + meta.json 이 같은 폴더에 있으면 자동으로 켜진다.  --no-eyelid 로 끈다.
출력:  captures/YYYYMMDD_HHMMSS/  face_measurement.jpg, measurement_screen.jpg,
                                  vision_landmarks.json, analysis_result.json, handoff.json(다음 단계로 넘길 것)

!! wflw_prep 부분(크롭 규칙·IMG·MARGIN)을 고치면 Colab 에서 모델을 다시 학습해야 한다.
!! face_metrics 부분을 고치면 (관상 카드를 쓸 경우) norms.json 을 다시 만들어야 한다.
"""

from __future__ import annotations

import math
import os
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import cv2


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
        pts = self.refiner(bgr, pts)                            # 2단계 눈 정밀 (모델 없으면 그대로)
        return add_virtual_points(pts)


# =============================================================================
#  [4b] eyelid - 쌍꺼풀 / 홑꺼풀 (선택).  Eyelid_Colab 이 만든 eyelid.tflite + meta.json 이 있으면 켜진다
# =============================================================================

EYELID_MODEL_FILE = 'eyelid.tflite'
EYELID_META_FILES = ('eyelid_meta.json', 'meta.json')
EYELID_EVERY = 3            # 검사를 통과한 프레임 3개마다 1번 판정 (모델이 무거워서)
EYELID_MAX_SAMPLES = 12     # 한 번의 스캔에서 이만큼 모이면 더 안 돌린다
EYELID_MIN_EYE_PX = 25      # 눈 너비(px)가 이보다 작으면 판정하지 않는다
EYELID_LOWRES_PX = 100      # 이보다 작으면 결과에 low_resolution 표시 (학습 사진보다 흐릿할 수 있음)

# 눈 크롭 규칙 - crop_eyes_v2.py 와 똑같아야 한다 (학습 때 자른 방식 == 지금 자르는 방식)
_PAIR_ASPECT, _PAIR_MARGIN, _PAIR_UP = 3.0, 1.45, 0.18
_SINGLE_ASPECT, _SINGLE_MARGIN, _SINGLE_UP = 1.5, 1.6, 0.12 / 0.667



def _eyelid_crop_oriented(img, center, u, v, cw, ch, out_size, flip=False):
    """center 를 중심으로 u,v 방향 cw x ch 영역을 회전 보정해서 자른다. 사진 밖으로 나가면 None"""
    cwi, chi = max(int(round(cw)), 8), max(int(round(ch)), 8)
    h, w = img.shape[:2]
    for sx, sy in ((-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)):
        p = center + u * (sx * cwi) + v * (sy * chi)
        if not (0 <= p[0] < w and 0 <= p[1] < h):
            return None
    t = center - u * (cwi / 2) - v * (chi / 2)
    M = np.array([[u[0], v[0], t[0]], [u[1], v[1], t[1]]], dtype=np.float64)
    crop = cv2.warpAffine(img, M, (cwi, chi), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)
    if flip:
        crop = cv2.flip(crop, 1)
    interp = cv2.INTER_AREA if cwi > out_size[0] else cv2.INTER_CUBIC
    return cv2.resize(crop, out_size, interpolation=interp)


def eyelid_eye_points(pts, width):
    """face_fortune 은 화면을 거울처럼 뒤집어서 점을 찾는다. 학습 사진은 뒤집지 않은 사진이므로 점의 x 를 되돌린다.
       되돌린 사진에서 왼쪽 눈 = 본인의 오른눈(R) = 우리 코드의 'right' 눈(72,68). {side: (바깥 눈초리, 안쪽 눈초리)}"""
    xy = np.asarray(pts, np.float64)[:, :2].copy()
    xy[:, 0] = (width - 1) - xy[:, 0]
    return {"R": (xy[EYES["right"]["outer"]], xy[EYES["right"]["inner"]]),
            "L": (xy[EYES["left"]["outer"]], xy[EYES["left"]["inner"]])}


def eyelid_crops(raw, eyes, size, pair):
    """뒤집지 않은 사진 + 눈 모서리 → ([크롭들], 눈 너비 px). 사진 밖으로 나가면 ([], 너비)"""
    ro, ri = eyes["R"]; lo, li = eyes["L"]
    rc, lc = (ro + ri) / 2, (lo + li) / 2
    roll = math.degrees(math.atan2(lc[1] - rc[1], lc[0] - rc[0]))
    a = math.radians(roll)
    u = np.array([math.cos(a), math.sin(a)]); v = np.array([-math.sin(a), math.cos(a)])
    wr, wl = float(np.linalg.norm(ro - ri)), float(np.linalg.norm(lo - li))
    if pair:
        cw = float(np.linalg.norm(ro - lo)) * _PAIR_MARGIN
        ch = cw / _PAIR_ASPECT
        crop = _eyelid_crop_oriented(raw, (rc + lc) / 2 - v * (_PAIR_UP * ch), u, v, cw, ch, size)
        return ([crop] if crop is not None else []), min(wr, wl)
    crops = []
    for center0, ew, flip in ((rc, wr, False), (lc, wl, True)):          # 왼눈은 좌우 반전해서 오른눈과 같은 방향으로
        cw = ew * _SINGLE_MARGIN
        ch = cw / _SINGLE_ASPECT
        c = _eyelid_crop_oriented(raw, center0 - v * (_SINGLE_UP * ch), u, v, cw, ch, size, flip)
        if c is not None:
            crops.append(c)
    return (crops if len(crops) == 2 else []), min(wr, wl)


def eyelid_unavailable(reason, status="unavailable", n=0):
    return {"status": status, "label": None, "probabilities": None, "n_samples": n, "validated": False,
            "source": "eyelid.tflite (ours)", "reason": reason}


class EyelidClassifier:
    """쌍꺼풀/홑꺼풀.  모델 파일이 없거나 열 수 없으면 ok=False 가 되어 이 기능만 꺼진다 (측정은 그대로)."""

    def __init__(self, enabled: bool = True) -> None:
        self.ok, self.reason, self.meta = False, "", {}
        if not enabled:
            self.reason = "turned off (--no-eyelid)"; return
        model = model_path(EYELID_MODEL_FILE)
        meta_file = next((model_path(n) for n in EYELID_META_FILES if os.path.exists(model_path(n))), None)
        if not os.path.exists(model) or meta_file is None:
            self.reason = "eyelid.tflite / meta.json not found next to face_fortune.py"; return
        try:
            self.meta = json.loads(Path(meta_file).read_text(encoding="utf-8"))
            self.it = _interpreter(model)
            self.inp = self.it.get_input_details()[0]
            self.out = self.it.get_output_details()[0]
            self.size = (int(self.meta["img_w"]), int(self.meta["img_h"]))
            self.pair = self.size[0] > self.size[1] * 2                      # 양눈 한 장 / 눈 하나씩
            self.positive = self.meta["positive_class"]                       # 시그모이드 출력이 가리키는 클래스
            self.other = [c for c in self.meta["labels"] if c != self.positive][0]
            self.threshold = float(self.meta.get("threshold", 0.5))
            self.ok = True
        except Exception as error:                                            # 모델이 깨졌어도 촬영은 계속
            self.reason = f"could not load eyelid model: {str(error)[:120]}"

    def _p_positive(self, crop_bgr: np.ndarray) -> float:
        x = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)[None]
        self.it.set_tensor(self.inp["index"], x)
        self.it.invoke()
        return float(self.it.get_tensor(self.out["index"]).ravel()[0])

    def predict(self, frame_mirrored: np.ndarray, pts: np.ndarray) -> dict[str, float] | None:
        """거울 프레임 + 점 → {p, eye_w} 또는 None (눈이 너무 작거나 사진 밖)"""
        if not self.ok:
            return None
        raw = cv2.flip(frame_mirrored, 1)
        crops, eye_w = eyelid_crops(raw, eyelid_eye_points(pts, frame_mirrored.shape[1]), self.size, self.pair)
        if not crops or eye_w < EYELID_MIN_EYE_PX:
            return None
        return {"p": float(np.mean([self._p_positive(c) for c in crops])), "eye_w": float(eye_w)}

    def aggregate(self, samples: list[dict[str, float]], final: bool = False) -> dict[str, Any]:
        if not self.ok:
            return eyelid_unavailable(self.reason)
        if len(samples) < 3:
            return eyelid_unavailable("collecting" if not final else "not enough valid samples (%d)" % len(samples),
                                      status="unavailable" if final else "pending", n=len(samples))
        p = float(np.mean([s["p"] for s in samples]))
        eye_w = float(np.median([s["eye_w"] for s in samples]))
        label = self.positive if p >= self.threshold else self.other           # 높은 쪽
        return {
            "status": "measured", "label": label,
            "probabilities": {self.positive: round(p, 4), self.other: round(1.0 - p, 4)},
            "n_samples": len(samples), "eye_width_px": round(eye_w, 1), "low_resolution": bool(eye_w < EYELID_LOWRES_PX),
            "threshold": self.threshold, "model": self.meta.get("model"),
            "model_test_accuracy": self.meta.get("test_accuracy", self.meta.get("cv_accuracy")),
            "validated": False,
            "source": "eyelid.tflite (ours; eye crops from WFLW eye corners)",
            "reason": "accuracy not validated on this camera; crops come from a different landmark model than training",
        }


# =============================================================================
#  [4c] labels - '관상 정보 총정리' 문서에서 정면 사진으로 잴 수 있는 것만: 5단계 라벨 + 가장 닮은 눈모양
# =============================================================================
#
#  눈꼬리 각도와 두 눈 사이 거리는 사용자가 정한 고정 표(TILT_SCALE / GAP_SCALE)로 5단계를 나눈다.
#  그 밖의 항목은 문서의 두 가지 기준으로 라벨을 붙인다 (기준 자료가 없는 항목은 라벨을 붙이지 않는다):
#   (1) 한국 성인 계측 논문의 평균 ± 표준편차  → z 점수로 5단계 (경계 z = -0.84, -0.25, 0.25, 0.84 → 각 20%)
#   (2) 고정 비율 기준                         → 인중: 코밑~윗입술 산 : 아랫입술 아래~턱끝 = 1:2 가 이상, 1:1 에 가까울수록 긴 인중
#  모든 비율은 face_metrics 의 IPD 정규화 값(feature_values_ipd)에서 다시 계산한다.
#  !! 눈 열림·눈꼬리·코 너비는 논문과 우리 점 모델의 측정 정의가 달라서 '참고' 이다 (아래 caveat 로 JSON 에도 남는다).

Z_CUTS = (-0.84, -0.25, 0.25, 0.84)
PHILTRUM_CUTS = (0.33, 0.40, 0.60, 0.80)          # 인중 ÷ (아랫입술 아래~턱끝).  이상 0.5(1:2), 1.0(1:1)=긴 인중. 문서 샘플 1:2.83(0.35)=짧음
PHILTRUM_IDEAL = 0.5

# ---- 눈꼬리 각도 5단계 (사용자 기준표, 한국인 평균 8.5° 중심) ----
#  각도 = 눈 안쪽 눈초리 → 바깥 눈초리 선의 기울기. 바깥이 더 높으면 +.  왼쪽·오른쪽 눈 중 '더 큰 값'으로 판정한다
TILT_AVG, TILT_SD = 8.5, 2.4                      # z 점수(닮은 눈모양 비교용)에만 쓴다. SD 는 문서의 논문값
TILT_CUTS = (5.0, 7.4, 9.7, 12.0)                 # 1단계 <5.0 | 2단계 5.0~7.3 | 3단계 7.4~9.6 | 4단계 9.7~12.0 | 5단계 >12.0
TILT_NAMES = ['매우 낮음', '낮음', '보통', '높음', '매우 높음']
TILT_NAMES_EN = ['VERY LOW', 'LOW', 'AVERAGE', 'HIGH', 'VERY HIGH']
TILT_SCALE = [   # (단계, 이름, 범위, 설명)
    (1, '매우 낮음', '5.0° 미만', '눈꼬리가 평균보다 많이 덜 올라간 편'),
    (2, '낮음', '5.0° ~ 7.3°', '평균보다 덜 올라간 편'),
    (3, '보통 (기준)', '7.4° ~ 9.6°', '한국인 평균(약 8.5°) 범위'),
    (4, '높음', '9.7° ~ 12.0°', '평균보다 더 올라간 편'),
    (5, '매우 높음', '12.0° 초과', '눈꼬리가 평균보다 많이 올라간 편'),
]

# ---- 두 눈 사이 거리 5단계 (사용자 기준표) ----
#  비율 = 두 눈 사이 거리(안쪽 눈초리 사이) ÷ 눈 가로 길이.  한국인 20~30대 평균 1.38 을 3단계(표준) 중심으로
GAP_AVG = 1.38
GAP_CUTS = (1.15, 1.30, 1.45, 1.60)               # 1단계 <1.15 | 2단계 1.15~<1.30 | 3단계 1.30~1.45 | 4단계 >1.45~1.60 | 5단계 >1.60
GAP_NAMES = ['매우 좁음', '약간 좁음', '표준', '약간 넓음', '매우 넓음']
GAP_NAMES_EN = ['VERY CLOSE', 'CLOSE', 'STANDARD', 'WIDE', 'VERY WIDE']
GAP_SCALE = [
    (1, '매우 좁음', '1.15 미만', '미간이 눈 크기에 비해 매우 좁은 편 (강한 구심형)'),
    (2, '약간 좁음', '1.15 ~ 1.30 미만', '평균보다 약간 좁으나 자연스러운 범위'),
    (3, '표준 (평균)', '1.30 ~ 1.45', '한국인 성인 평균 범위 (약 1.38)'),
    (4, '약간 넓음', '1.45 초과 ~ 1.60', '평균보다 약간 넓어 순하고 부드러운 인상'),
    (5, '매우 넓음', '1.60 초과', '미간이 눈 크기에 비해 매우 넓은 편 (원심형)'),
]


# ---- 눈썹 각도 (사용자 기준): 0° 미만 아래로 내려감(팔자) / 0~5° 평탄(일자형) / 5~12° 평균 / 12° 이상 치켜 올라감 ----
#  각도 = 눈썹 안쪽 → 바깥쪽 선의 기울기, 바깥이 높으면 +.  양쪽 눈썹의 평균 (face_metrics 의 brow_tilt)
BROW_CUTS = (0.0, 5.0, 12.0)
BROW_NAMES = ['내려간(팔자)형', '일자형', '보통', '', '위로 솟은형']          # 4번째 칸은 쓰지 않는다 (표가 4구간)
BROW_NAMES_EN = ['DOWNTURNED', 'STRAIGHT', 'AVERAGE', '', 'RAISED']

# ---- 코 길이 (얼굴에 비해) = 코 길이(코뿌리~코밑) ÷ 광대 폭.  임시 기준 (평균 약 0.35) ----
NOSE_CUTS = (0.30, 0.33, 0.37, 0.40)
NOSE_NAMES = ['아주 짧은 편', '짧은 편', '보통', '긴 편', '아주 긴 편']
NOSE_NAMES_EN = ['VERY SHORT', 'SHORT', 'AVERAGE', 'LONG', 'VERY LONG']


# ---- 얼굴 가로세로 비율 (세로 ÷ 가로) = face_metrics 의 face_ratio.  한국 여성 평균 1.38 을 '보통' 가운데로.  임시 기준 ----
FACE_AVG = 1.38
FACE_CUTS = (1.15, 1.30, 1.45, 1.60)
FACE_NAMES = ['아주 둥근 편', '둥근 편', '보통', '긴 편', '아주 긴 편']
FACE_NAMES_EN = ['VERY ROUND', 'ROUND', 'AVERAGE', 'LONG', 'VERY LONG']


def face_level(ratio):
    return int(ratio >= FACE_CUTS[0]) + int(ratio >= FACE_CUTS[1]) + int(ratio > FACE_CUTS[2]) + int(ratio > FACE_CUTS[3])


# ---- 입술 (한국인 평균 자료: 윗입술 7.8 mm · 아랫입술 12.2 mm [Ioi 외], 입 너비 약 45 mm [Kwon 외·한국 청년 251명]) ----
#  두께 = (윗입술 + 아랫입술 입술살 두께) ÷ 입 너비.  평균 약 (7.8+12.2)/45 = 0.44
LIP_CUTS = (0.32, 0.38, 0.50, 0.56)
LIP_NAMES = ['아주 얇은 편', '얇은 편', '보통', '도톰한 편', '아주 도톰한 편']
LIP_NAMES_EN = ['VERY THIN', 'THIN', 'AVERAGE', 'FULL', 'VERY FULL']
#  균형 = 윗입술 ÷ 아랫입술.  한국인 평균 0.64 (7.8/12.2) ~ 0.90 (한국 여성 1:1.11), 선호 0.8.  0.65~1.0 을 '균형'으로
LIPBAL_CUTS = (0.65, 1.0)
LIPBAL_NAMES = ['아랫입술이 도톰', '', '균형', '', '윗입술이 도톰']
LIPBAL_NAMES_EN = ['LOWER FULLER', '', 'BALANCED', '', 'UPPER FULLER']
#  입꼬리 = face_metrics 의 mouth_tilt (입술 한가운데 → 양쪽 입꼬리 선의 기울기, 올라가면 +).  한국인 평균 자료가 없어서
#  입술 구조로 계산한 임시값: 힘 뺀 입은 약 +5° 라서 2°~9° 를 '보통'으로
CORNER_CUTS = (2.0, 9.0)
CORNER_NAMES = ['처진 편', '', '보통', '', '올라간 편']
CORNER_NAMES_EN = ['DOWNTURNED', '', 'AVERAGE', '', 'UPTURNED']
MOUTH_OPEN_GAP = 0.08          # 윗·아랫입술 안쪽 사이가 동공 거리의 8% 넘게 벌어지면 입이 벌어진 것으로 보고 라벨에 ? 표시


# ---- 그 밖에 잴 수 있는 부위 (기준표 face_reference*.csv 가 있으면 그 백분위로 20%씩, 없으면 아래 임시 평균±편차로) ----
#  key: (기준표에서 찾을 측정값, 임시 판정에 쓸 값을 만드는 함수, 임시 평균, 임시 편차, 단계 이름 5개, 영어 5개, 설명 형식)
_SIZE5 = (['아주 작은 편', '작은 편', '보통', '큰 편', '아주 큰 편'], ['VERY SMALL', 'SMALL', 'AVERAGE', 'LARGE', 'VERY LARGE'])
_LEN5 = (['아주 짧은 편', '짧은 편', '보통', '긴 편', '아주 긴 편'], ['VERY SHORT', 'SHORT', 'AVERAGE', 'LONG', 'VERY LONG'])
_WIDE5 = (['아주 좁은 편', '좁은 편', '보통', '넓은 편', '아주 넓은 편'], ['VERY NARROW', 'NARROW', 'AVERAGE', 'WIDE', 'VERY WIDE'])
FLEX = {
    'forehead': ('upper_cheek_ratio', lambda m: m['upper_cheek_ratio'], 0.75, 0.05, _WIDE5, "forehead/cheek %.2f"),
    'cheek': ('cheek_w', lambda m: m['cheek_w'], 2.20, 0.12, _WIDE5, "cheek/IPD %.2f"),
    'eye_size': ('eye_len', lambda m: m['eye_len'], 0.42, 0.03, _SIZE5, "eye/IPD %.3f"),
    'brow_len': ('brow_len', lambda m: m['brow_len'] / max(m['eye_len'], 1e-6), 1.60, 0.20, _LEN5, "brow/eye %.2f (1 이상이면 눈보다 김)"),
    'brow_eye_gap': ('brow_eye_gap', lambda m: m['brow_eye_gap'] / max(m['eye_len'], 1e-6), 0.65, 0.12, _WIDE5, "brow-eye gap/eye %.2f"),
    'glabella': ('glabella_w', lambda m: m['glabella_w'] / max(m['eye_len'], 1e-6), 1.40, 0.20, _WIDE5, "glabella/eye %.2f"),
}
SYM_NAMES = ['고른 편', '', '보통', '', '치우친 편']
SYM_NAMES_EN = ['BALANCED', '', 'AVERAGE', '', 'UNEVEN']
SYM_FALLBACK = (0.045, 0.015)           # 좌우 대칭 오차의 임시 평균·편차 (작을수록 고름)


def _flex_item(key, m, ref):
    """기준표가 있으면 그 측정값의 백분위로 20%씩, 없으면 임시 평균·편차의 z 점수로 5단계"""
    metric, value_fn, mean, sd, (names, names_en), fmt = FLEX[key]
    value = float(value_fn(m))
    if ref is not None and metric in m and ref.has(metric):
        pct = ref.percentile(metric, m[metric])
        item = _fixed_item(key, sum(pct >= c for c in (20, 40, 60, 80)), value, _z_from_pct(pct), (fmt + " (p%d)") % (value, pct), names, names_en, "reference_table")
        item["percentile"] = round(pct, 1)
        return item
    z = (value - mean) / sd
    return _fixed_item(key, sum(z >= c for c in Z_CUTS), value, z, (fmt + " (임시 평균 %.2f)") % (value, mean), names, names_en, "provisional_mean_sd")


def _symmetry_item(m, ref):
    v = float(m['symmetry'])
    if ref is not None and ref.has('symmetry'):
        pct = ref.percentile('symmetry', v)
        idx = 0 if pct <= 30 else (4 if pct >= 80 else 2)
        return _fixed_item('symmetry', idx, v, _z_from_pct(pct), "asymmetry %.3f (p%d, 작을수록 고름)" % (v, pct), SYM_NAMES, SYM_NAMES_EN, "reference_table")
    z = (v - SYM_FALLBACK[0]) / SYM_FALLBACK[1]
    idx = 0 if z < -0.5 else (4 if z > 1.0 else 2)
    return _fixed_item('symmetry', idx, v, z, "asymmetry %.3f (임시 평균 %.3f, 작을수록 고름)" % (v, SYM_FALLBACK[0]), SYM_NAMES, SYM_NAMES_EN, "provisional_mean_sd")


# ---- 운세 7가지: 라벨 단계(1~5)를 관상의 부위-운 연결(십이궁 등)대로 모아 '부족 / 보통 / 좋음' 으로 ----
#  (라벨, 가중치, 방향)  방향 +1: 단계가 높을수록 좋음 / -1: 낮을수록 좋음 / 0: 가운데(균형)일수록 좋음
LUCK_RULES = {
    'wealth': ('재물운', [('nose_len', 1.0, 1), ('lip_thick', 1.0, 1), ('brow_eye_gap', 1.0, 1), ('mouth_w', 0.5, 1), ('face_ratio', 0.5, -1)]),
    'love': ('애정운 (연애운)', [('eye_open', 1.0, 1), ('lip_thick', 1.0, 1), ('mouth_corner', 1.0, 1), ('eye_size', 0.5, 1), ('lip_balance', 0.5, 0)]),
    'health': ('건강운 (생명력)', [('philtrum', 1.0, 1), ('symmetry', 1.0, -1), ('nose_len', 0.5, 1), ('cheek', 0.5, 1)]),
    'career': ('직업운 (명예 · 관록)', [('forehead', 1.0, 1), ('cheek', 1.0, 1), ('glabella', 0.5, 1), ('eye_size', 0.5, 1)]),
    'people': ('인간관계운 (인복)', [('brow_len', 1.0, 1), ('glabella', 1.0, 1), ('mouth_corner', 1.0, 1), ('eye_gap', 0.5, 1)]),
    'family': ('가정운 (집 · 부동산 · 자녀)', [('brow_eye_gap', 1.0, 1), ('philtrum', 1.0, 1), ('lip_balance', 0.5, 0)]),
    'late': ('말년운', [('philtrum', 1.0, 1), ('mouth_corner', 1.0, 1), ('lip_thick', 0.5, 1)]),
}


def luck_report(items):
    """라벨 → 운세 7가지.  점수 -1 ~ 1 (부족 < -0.25 ≤ 보통 ≤ 0.25 < 좋음), 무엇이 끌어올리고 끌어내렸는지도 같이"""
    out = []
    for key, (name, parts) in LUCK_RULES.items():
        total = wsum = 0.0
        ups, downs = [], []
        for lab, w, d in parts:
            it = items.get(lab)
            if it is None:
                continue
            lv = it['level']
            c = (1.0 - abs(lv - 3)) if d == 0 else d * (lv - 3) / 2.0
            total += w * c
            wsum += w
            (ups if c > 0 else downs if c < 0 else []).append('%s %s' % (it['name_ko'].split(' (')[0], it['label']))
        score = total / wsum if wsum else 0.0
        state = 'low' if score < -0.25 else ('high' if score > 0.25 else 'mid')
        out.append({"key": key, "name": name, "score": round(score, 2), "state": state, "ups": ups, "downs": downs})
    return out


def brow_level(deg):
    """눈썹 각도(°) → 0(내려감) 1(일자) 2(평균) 4(치켜 올라감)"""
    return 0 if deg < BROW_CUTS[0] else (1 if deg < BROW_CUTS[1] else (2 if deg < BROW_CUTS[2] else 4))


def nose_level(ratio):
    return sum(ratio >= c for c in NOSE_CUTS)


def tilt_level(deg):
    """눈꼬리 각도(°) → 0~4 (표의 1~5단계)"""
    return int(deg >= TILT_CUTS[0]) + int(deg >= TILT_CUTS[1]) + int(deg >= TILT_CUTS[2]) + int(deg > TILT_CUTS[3])


def gap_level(ratio):
    """두 눈 사이 ÷ 눈 가로 길이 → 0~4 (표의 1~5단계)"""
    return int(ratio >= GAP_CUTS[0]) + int(ratio >= GAP_CUTS[1]) + int(ratio > GAP_CUTS[2]) + int(ratio > GAP_CUTS[3])

def _pooled(a, b):
    (ma, sa), (mb, sb) = a, b
    return ((ma + mb) / 2, math.sqrt((sa ** 2 + sb ** 2) / 2 + ((ma - mb) / 2) ** 2))

LABEL_REF = {
    'eye_w_gap': (0.70, 0.07),        # 눈 너비 ÷ 두 눈 사이(안쪽 눈초리)   (Park 외 · Kwon 외)
    'eye_open':  (0.305, 0.0455),     # 눈 세로 ÷ 가로: 남 0.30±0.04, 여 0.31±0.05 를 합침   [참고: 점 모델마다 오프셋이 다름]
    'eye_tilt':  (8.4, 2.4),          # 눈꼬리 기울기(°): 남 7.9±2.4, 여 8.8±2.3 을 합침     [참고: 측정 정의가 다름]
    'mouth_ipd': (0.70, 0.06),        # 입 너비 ÷ 동공 사이 (여성 자료. 남성 자료는 없음)
    'nose_gap': {'f': (1.09, 0.11), 'm': (1.02, 0.10)},   # 코 너비 ÷ 두 눈 사이 (남녀 차이 큼)  [참고]
}
LABEL_WORDS = {   # (낮을 때, 높을 때) 한글 / 영어
    'eye_tilt':  (('처진', '올라간'), ('DOWNTURNED', 'UPTURNED')),
    'eye_open':  (('가는', '둥근'), ('NARROW', 'ROUND')),
    'eye_gap':   (('좁은', '먼'), ('CLOSE-SET', 'WIDE-SET')),
    'nose_w':    (('좁은', '넓은'), ('NARROW', 'WIDE')),
    'mouth_w':   (('작은', '큰'), ('SMALL', 'LARGE')),
    'philtrum':  (('짧은', '긴'), ('SHORT', 'LONG')),
}
LABEL_NAME = {
    'eye_tilt': ('눈꼬리 각도', 'Eye tilt'), 'eye_open': ('눈 열림 (세로/가로)', 'Eye opening'),
    'lip_thick': ('입술 두께 (÷입 너비)', 'Lip thickness'), 'lip_balance': ('윗·아랫입술 균형', 'Lip balance'), 'mouth_corner': ('입꼬리', 'Mouth corner'),
    'face_ratio': ('얼굴 가로세로 비율', 'Face ratio'), 'forehead': ('이마 폭 (관록궁)', 'Forehead'), 'cheek': ('광대 폭', 'Cheekbone'),
    'symmetry': ('좌우 균형', 'Symmetry'), 'eye_size': ('눈 크기', 'Eye size'), 'brow_len': ('눈썹 길이 (형제궁)', 'Brow length'),
    'brow_eye_gap': ('눈썹과 눈 사이 (전택궁)', 'Brow-eye gap'), 'glabella': ('미간 (명궁 · 인당)', 'Glabella'), 'eye_gap': ('두 눈 사이 (÷눈 길이)', 'Eye gap'), 'brow_tilt': ('눈썹 각도', 'Brow tilt'), 'nose_len': ('코 길이 (얼굴 대비)', 'Nose length'), 'nose_w': ('코 너비', 'Nose width'),
    'mouth_w': ('입 크기', 'Mouth size'), 'philtrum': ('인중 길이', 'Philtrum'),
}
LABEL_ORDER = ['face_ratio', 'forehead', 'cheek', 'symmetry', 'eye_tilt', 'eye_open', 'eye_size', 'eye_gap', 'brow_tilt', 'brow_len', 'brow_eye_gap',
               'glabella', 'nose_len', 'mouth_w', 'lip_thick', 'lip_balance', 'mouth_corner', 'philtrum']   # 코 '너비'는 뺐고 코 '길이'를 넣었다
LABEL_CAVEAT = {'forehead': '참고: 헤어라인 모델이 없으면 눈썹 바깥 끝으로 대신 잰 값 · 기준표가 없으면 임시 평균', 'cheek': '참고: 기준표가 없으면 임시 평균',
                'symmetry': '참고: 기준표가 없으면 임시 평균', 'eye_size': '참고: 한국인 평균 눈 길이(약 27 mm ÷ 동공 거리 64 mm)', 'brow_len': '참고: 기준표가 없으면 임시 평균',
                'brow_eye_gap': '참고: 기준표가 없으면 임시 평균', 'glabella': '참고: 기준표가 없으면 임시 평균',
                'lip_thick': '참고: 한국인 평균(윗 7.8 + 아랫 12.2 mm, 입 너비 약 45 mm)에서 계산한 임시 기준', 'lip_balance': '참고: 한국인 평균 윗/아랫 0.64~0.90 에서 정한 임시 기준',
                'mouth_corner': '참고: 한국인 평균 자료가 없어 구조로 계산한 임시 기준 (힘 뺀 무표정에서 재야 해요)',
                'face_ratio': '참고: 이마 상단 정의가 논문과 달라 기준이 임시값',
                'brow_tilt': '참고: 각도 기준은 사용자 표이고 점 모델에 따라 달라질 수 있음', 'nose_len': '참고: 비율 기준이 임시값 (데이터로 조정 필요)',
                'eye_tilt': '참고: 기준표와 우리 점 모델의 측정 정의가 다를 수 있음', 'eye_gap': '참고: 기준표와 우리 점 모델의 측정 정의가 다를 수 있음',
                'eye_open': '참고: 점 모델마다 오프셋이 다름 (보정 필요)',
                'nose_w': '참고: 코 점 정의가 논문과 다름', 'mouth_w': '여성 평균 기준 (남성 자료 없음)'}
LEVEL_EN = ['VERY {}', '{}', 'AVERAGE', '{}', 'VERY {}']
LEVEL_COLORS = [(255, 175, 100), (255, 205, 150), (205, 205, 215), (120, 195, 255), (80, 150, 255)]   # BGR


def _label_item(key, level, value, z, caption, basis):
    (lo_ko, hi_ko), (lo_en, hi_en) = LABEL_WORDS[key]
    ko = ('보통' if level == 2 else LEVELS[level] % (lo_ko if level < 2 else hi_ko))
    en = ('AVERAGE' if level == 2 else LEVEL_EN[level].format(lo_en if level < 2 else hi_en))
    item = {"name_ko": LABEL_NAME[key][0], "name_en": LABEL_NAME[key][1], "label": ko, "label_en": en,
            "level": level + 1, "of": 5, "value": round(float(value), 4),
            "z": None if z is None else round(float(z), 3), "caption": caption, "basis": basis}
    if key in LABEL_CAVEAT:
        item["caveat"] = LABEL_CAVEAT[key]
    return item


def _fixed_item(key, level, value, z, caption, names, names_en, basis):
    """고정 표(눈꼬리 각도 · 두 눈 사이)로 만든 라벨. 단계 이름은 표의 이름을 그대로 쓴다"""
    item = {"name_ko": LABEL_NAME[key][0], "name_en": LABEL_NAME[key][1], "label": names[level], "label_en": names_en[level],
            "level": level + 1, "of": 5, "value": round(float(value), 4),
            "z": None if z is None else round(float(z), 3), "caption": caption, "basis": basis}
    if key in LABEL_CAVEAT:
        item["caveat"] = LABEL_CAVEAT[key]
    return item


# ---- 가장 닮은 눈모양: 문서의 16가지 중 '모양으로 잴 수 있는 조건'만 쓴다 ----
# 눈동자 색·크기, 흰자 노출, 눈빛, 안구 돌출, 눈 주변 색은 우리 점 모델(눈동자 중심만 있음)로 못 재서 비교에서 뺀다.
# 그래서 이리눈·사백안·뱀눈·돼지눈·게눈·물고기눈(흰자·눈동자 기준)과 사슴눈·거북이눈·원숭이눈(눈동자 기준)은 후보가 아니다.
# 목표 값은 z 점수(0=평균). tilt=눈꼬리(+ 올라감), open=눈 열림(+ 둥글), size=눈 크기(+ 큼), lid=쌍꺼풀 필요 여부
EYE_TYPES = [
    # 이름, 영어, 목표 {특징: (z, 가중치)}, 쌍꺼풀 필요, 한 줄 설명 (문서의 속설을 순화)
    ('봉황눈', 'Phoenix', dict(tilt=(1.3, 2.0), open=(-1.0, 1.5), size=(0.8, 1.0)), False, '총명하고 뜻한 바를 이루는 인상'),
    ('용눈', 'Dragon', dict(tilt=(0.6, 1.5), open=(0.2, 0.5), size=(1.3, 2.0)), False, '리더십과 결단력이 느껴지는 인상'),
    ('학눈', 'Crane', dict(tilt=(0.0, 0.7), open=(-0.8, 1.2), size=(1.0, 1.5)), False, '맑고 이상이 높은 인상'),
    ('코끼리눈', 'Elephant', dict(tilt=(0.0, 0.5), open=(-1.0, 1.5), size=(0.8, 1.2)), True, '온화하고 인자한 지도자 인상'),
    ('호랑이눈', 'Tiger', dict(tilt=(0.3, 0.7), open=(0.6, 1.2), size=(1.3, 2.0)), False, '정의감과 위엄이 느껴지는 인상'),
    ('소눈', 'Ox', dict(tilt=(-0.3, 0.7), open=(1.2, 2.0), size=(0.8, 1.2)), True, '인내심 있고 부지런한 실속형 인상'),
    ('원앙눈', 'Mandarin duck', dict(tilt=(-1.3, 2.5), open=(0.0, 0.5), size=(-0.3, 0.5)), False, '인기와 매력, 예술적 적성이 있는 인상'),
]
EYE_NOT_MEASURABLE = ['눈동자 색·크기', '흰자 노출', '눈빛·충혈', '안구 돌출', '눈 주변 색']
EYE_DISTINCT_MIN = 0.40        # 유사도가 이보다 낮으면 '뚜렷한 유형 없음' (평균적인 눈은 어느 유형과도 0%에 가깝다)


def eye_shape_match(z_tilt, z_open, z_size, lid_label):
    """측정된 z 점수와 각 눈모양의 목표를 비교해 닮은 순서로 돌려준다. lid_label: 'double' / 'single' / None(모름)"""
    feats = {'tilt': z_tilt, 'open': z_open, 'size': z_size}
    ranking = []
    for ko, en, target, needs_double, desc in EYE_TYPES:
        # 유사도 = 1 - (목표와의 오차 에너지) / (목표 자체의 에너지).  평균적인 눈(z=0)은 모든 유형과 0 이 되고,
        # 목표와 정확히 같은 눈은 1, 반대 방향이면 0 으로 잘린다 (R² 와 같은 꼴)
        err = tgt = 0.0
        for f, (t, w) in target.items():
            err += w * (max(-3.0, min(3.0, feats[f])) - t) ** 2
            tgt += w * t * t
        if needs_double and lid_label in ('double', 'single'):          # 쌍꺼풀 판정이 있을 때만 반영 (+1 쌍꺼풀 / -1 홑꺼풀)
            err += 1.5 * ((1.0 if lid_label == 'double' else -1.0) - 1.0) ** 2
            tgt += 1.5
        raw = 1.0 - err / max(tgt, 1e-6)
        ranking.append({"name": ko, "name_en": en, "similarity": round(max(0.0, raw), 3), "description": desc, "_raw": raw})
    ranking.sort(key=lambda r: -r["_raw"])      # 0 으로 잘린 동점이어도 더 가까운 쪽이 앞에 오게
    for r in ranking:
        r.pop("_raw")
    return ranking


# ---- 기준표 (face_reference*.csv): 우리 점 모델 + IPD 정규화로 잰 42개 측정값의 분포 ----
# 열: feature, mean, std, min, p10 … p90, max, count.  문서의 논문 평균은 MediaPipe 점으로 잰 값이라 우리 점 모델과 정의가 달라서
# (예: 입 너비 ÷ 동공 사이가 논문 0.70 vs 우리 0.80) 그대로 쓰면 평균적인 얼굴이 전부 '아주 ~한 편'으로 나온다. 이 표가 그 문제를 없앤다.
REF_PCTS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
REF_LEVEL_CUTS = (20, 40, 60, 80)              # 백분위 20% 씩 5단계
REF_FILE_NAMES = ('face_reference_10level_clean.csv', 'face_reference.csv')


class FaceReference:
    def __init__(self, path):
        import csv
        self.path = str(path)
        self.stats: dict[str, dict[str, Any]] = {}
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                knots = [float(row[c]) for c in ("min", "p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90", "max")]
                knots = list(np.maximum.accumulate(np.array(knots) + np.arange(len(knots)) * 1e-12))   # 같은 값이 이어져도 보간이 되게
                self.stats[row["feature"]] = {"mean": float(row["mean"]), "std": float(row["std"]),
                                              "knots": knots, "count": int(float(row["count"]))}
        if not self.stats:
            raise ValueError("empty reference table")

    def has(self, *features):
        return all(f in self.stats for f in features)

    def percentile(self, feature, value):
        """값이 기준 분포에서 몇 백분위인지 (0~100). 분포 밖은 0 / 100 으로 잘림"""
        return float(np.interp(value, self.stats[feature]["knots"], REF_PCTS))

    def median(self, feature):
        return self.stats[feature]["knots"][5]

    def std(self, feature):
        return self.stats[feature]["std"]


def find_reference(explicit=None):
    """--reference 로 준 파일, 없으면 이 파일 옆의 face_reference*.csv (브라우저가 붙인 '(1)' 같은 이름도 허용)"""
    cands = [Path(explicit)] if explicit else []
    cands += [Path(model_path(n)) for n in REF_FILE_NAMES]
    cands += sorted(Path(_MODEL_DIR).glob("face_reference*.csv"))
    for c in cands:
        if c.exists():
            try:
                return FaceReference(c)
            except Exception as error:
                print("reference table could not be read:", c, error)
    return None


def _z_from_pct(pct):
    from statistics import NormalDist
    return NormalDist().inv_cdf(min(max(pct, 0.5), 99.5) / 100.0)


def _reliability(ref, feature, std, n):
    """30프레임 평균의 흔들림(표준오차)이 사람 사이의 차이(표준편차)에 비해 얼마나 큰지. 크면 라벨이 한 칸 흔들릴 수 있다"""
    if not std or feature not in std or not n or n < 5:
        return None, None
    ratio = (std[feature] / math.sqrt(n)) / max(ref.std(feature), 1e-9)
    return ("low" if ratio > 0.5 else "ok"), round(float(ratio), 2)


def _ref_item(ref, key, feature, value, shown, fmt, std, n, invert=False):
    pct = ref.percentile(feature, value)
    if invert:
        pct = 100.0 - pct
    level = sum(pct >= c for c in REF_LEVEL_CUTS)
    item = _label_item(key, level, shown, _z_from_pct(pct), "", "reference_table_quintile")
    item["percentile"] = round(pct, 1)
    item["caption"] = fmt % (shown, int(round(pct)), (1.0 / ref.median(feature)) if invert and feature == 'eye_aspect' else ref.median(feature))
    rel, ratio = _reliability(ref, feature, std, n)
    if rel:
        item["reliability"], item["noise_ratio"] = rel, ratio
    item.pop("caveat", None)
    return item


# =============================================================================
#  규칙집 (gwansang_rules.json, 이 파일 옆에 둔다): 오행 얼굴형 · 삼정 · 십이궁 · 운세 5가지 · 종합 등급
# =============================================================================
#  얼굴형·삼정은 측정값만으로 정한다.  십이궁·운세·특징·등급은 '백분위'가 필요해서 기준표(face_reference*.csv)가 있어야 계산된다.
#  규칙집의 지표 중 이 점 모델에 없는 것(bridge_w · forehead_tilt · nose_tip_h)은 건너뛰고, 있는 지표만으로 평균을 낸다.

RULES_FILE = 'gwansang_rules.json'
_RULES: dict[str, Any] = {}
ZONE_AVG, ZONE_TOL = 100.0 / 3, 2.0                  # 삼정: 세 구역이 각 33.3% 가 균형, ±2%p 안이면 균형 (임시)
TRAIT_COVERED = {'eye_aspect', 'eye_gap', 'brow_tilt', 'mouth_w', 'lip_up', 'philtrum', 'eye_tilt', 'nose_tip_h'}   # 이미 라벨이 있는 것은 뺀다
TRAIT_PCT = (15, 85)                                  # 이 백분위 이하/이상일 때만 '눈에 띄는 특징'으로


def load_rules():
    """gwansang_rules.json → dict.  없거나 깨졌으면 None (이 기능만 꺼진다)"""
    if 'v' not in _RULES:
        try:
            _RULES['v'] = json.loads(Path(model_path(RULES_FILE)).read_text(encoding='utf-8'))
        except Exception as error:
            _RULES['v'], _RULES['err'] = None, str(error)[:80]
    return _RULES['v']


def ohaeng_key(m):
    """오행 얼굴형 → 'square'(금) / 'round'(수) / 'oblong'(목) / 'heart'(화) / 'oval'(토).  긴 얼굴 기준은 라벨(FACE_CUTS)과 맞춘다"""
    r, fw, cw, jw, ang = m['face_ratio'], m['forehead_w'], m['cheek_w'], m['jaw_w'], m['jaw_angle']
    if r > FACE_CUTS[2]:
        return 'oblong'
    if fw > jw * 1.12 and r > FACE_CUTS[1]:
        return 'heart'
    if ang < 118 and jw > cw * 0.82:
        return 'square'
    if r < FACE_CUTS[1]:
        return 'round'
    return 'oval'


def _score(ref, m, metrics):
    """[(지표, 'high'|'low')] → 0~100 점수 (high 는 백분위 그대로, low 는 뒤집어서 평균).  쓸 수 있는 지표가 없으면 None"""
    vals = []
    for name, direction in metrics:
        if ref is not None and name in m and ref.has(name):
            p = ref.percentile(name, m[name])
            vals.append(p if direction == 'high' else 100.0 - p)
    return float(np.mean(vals)) if vals else None


def rules_report(m, ref):
    rules = load_rules()
    if rules is None:
        return {"status": "no_rules", "reason": "%s 를 이 파일 옆에 두면 켜져요 (%s)" % (RULES_FILE, _RULES.get('err', ''))}
    out = {"status": "ok" if ref is not None else "no_reference", "disclaimer_rules": rules.get("disclaimer", "")}
    key = ohaeng_key(m)
    out["ohaeng"] = dict(rules['ohaeng'][key], key=key)
    zones = []                                                             # 삼정
    for zk in ('zone_top', 'zone_mid', 'zone_bot'):
        z, v = rules['zones'][zk], float(m[zk])
        state = 'short' if v < ZONE_AVG - ZONE_TOL else ('long' if v > ZONE_AVG + ZONE_TOL else 'even')
        zones.append({"key": zk, "name": z['name'], "period": z['period'], "value": round(v, 1), "state": state,
                      "text": z['short'] if state == 'short' else (z['long'] if state == 'long' else z['intro'])})
    out["zones"] = zones
    if ref is None:
        out["reason"] = "face_reference*.csv 가 있어야 십이궁 · 운세 · 특징 · 등급이 계산돼요"
        return out
    lv3 = lambda sc: 0 if sc < 100 / 3 else (1 if sc < 200 / 3 else 2)    # 글 3단계
    lv4 = lambda sc: 0 if sc < 25 else (1 if sc < 50 else (2 if sc < 75 else 3))   # 배지 4단계
    palaces = []
    for p in rules['palaces']:
        sc = _score(ref, m, p['metrics'])
        if sc is None:
            continue
        palaces.append({"key": p['key'], "name": p['name'], "hanja": p['hanja'], "region": p['region'], "score": round(sc, 1),
                        "badge": rules['palace_levels'][lv4(sc)], "badge_idx": lv4(sc), "level": lv3(sc) + 1, "text": p['text'][str(lv3(sc))]})
    out["palaces"] = palaces
    fortunes = []
    for f in rules['fortunes']:
        sc = _score(ref, m, f['metrics'])
        if sc is None:
            continue
        t = f['text'][str(lv3(sc))]
        fortunes.append({"key": f['key'], "name": f['name'], "parts": f['parts'], "score": round(sc, 1), "level": lv3(sc) + 1,
                         "title": t['title'], "body": t['body'], "caution": t['caution'], "advice": t['advice']})
    out["fortunes"] = fortunes
    traits = []                                                            # 눈에 띄는 특징 (이미 라벨이 있는 지표는 제외)
    for name, texts in rules['traits'].items():
        if name in TRAIT_COVERED or name not in m or not ref.has(name):
            continue
        p = ref.percentile(name, m[name])
        if p <= TRAIT_PCT[0] or p >= TRAIT_PCT[1]:
            traits.append({"metric": name, "name": METRIC_KO.get(name, name), "percentile": round(p, 1), "text": texts[0] if p <= TRAIT_PCT[0] else texts[1]})
    out["traits"] = sorted(traits, key=lambda t: -abs(t["percentile"] - 50))
    strong = sum(1 for p in palaces if p["level"] == 3)                    # 글 3단계 중 가장 높은 단계인 궁의 수
    grade = next(g for g in sorted(rules['grades'], key=lambda g: -g['min_strong']) if strong >= g['min_strong'])
    out["grade"] = dict(grade, strong=strong, total=len(palaces))
    return out


def eye_tilts(eyes_wflw):
    """analysis['eyes_wflw'] (또는 aggregate_eyes 결과) → {'left': 각도, 'right': 각도}"""
    try:
        return {s: float(eyes_wflw[s]['metrics']['outer_corner_tilt_deg']) for s in ('left', 'right')}
    except (KeyError, TypeError):
        return None


def face_labels(m, eyelid=None, sex='any', ref=None, std=None, n=None, tilts=None):
    """IPD 정규화 측정값 m (feature_values_ipd) → 5단계 라벨 + 닮은 눈모양. m 이 없으면 None.
       눈꼬리 각도 · 두 눈 사이 거리: 고정 표(TILT_SCALE / GAP_SCALE).  그 밖: ref(기준표 CSV)가 있으면 20%씩, 없으면 문서의 논문 평균±편차
       std, n: 프레임들의 표준편차 dict 와 프레임 수 (라벨이 흔들릴지 가늠하는 데 씀)"""
    if not m:
        return None
    eps = 1e-6
    use_ref = ref is not None and ref.has('eye_aspect', 'eye_len', 'mouth_w')
    eye_open = ((m['eye_h_l'] + m['eye_h_r']) / 2) / max(m['eye_len'], eps)  # 눈 세로 ÷ 가로
    phil = m['philtrum'] / max(m['chin_len'], eps)                           # 인중 ÷ (아랫입술 아래~턱끝)
    phil_item = _label_item('philtrum', sum(phil >= c for c in PHILTRUM_CUTS), phil, None,
                            "philtrum 1:%.1f (ideal 1:2)" % (1.0 / max(phil, eps)), "fixed_ratio_1_to_2")

    # 눈꼬리 각도 · 두 눈 사이 거리 - 사용자 고정 표
    tilt = max(tilts.values()) if tilts else float(m['eye_tilt'])      # 두 눈 중 더 큰 값 (각도를 못 받으면 평균으로)
    z_tilt = (tilt - TILT_AVG) / TILT_SD
    tilt_item = _fixed_item('eye_tilt', tilt_level(tilt), tilt, z_tilt, "%+.1f deg (큰 쪽 눈, 평균 %.1f)" % (tilt, TILT_AVG),
                            TILT_NAMES, TILT_NAMES_EN, "user_table_eye_tilt")
    gap_ratio = float(m['eye_gap']) / max(float(m['eye_len']), eps)
    gap_item = _fixed_item('eye_gap', gap_level(gap_ratio), gap_ratio, None, "gap/eye width %.2f (평균 %.2f, 표준 1.30~1.45)" % (gap_ratio, GAP_AVG),
                           GAP_NAMES, GAP_NAMES_EN, "user_table_eye_gap")
    brow_deg = float(m['brow_tilt'])
    brow_item = _fixed_item('brow_tilt', brow_level(brow_deg), brow_deg, None,
                            "%+.1f deg (<0 내려감 / 0~5 일자 / 5~12 평균 / 12~ 치켜)" % brow_deg, BROW_NAMES, BROW_NAMES_EN, "user_table_brow_tilt")
    face_r = float(m['face_ratio'])
    face_item = _fixed_item('face_ratio', face_level(face_r), face_r, None,
                            "face length/width %.2f (평균 %.2f, 임시 기준)" % (face_r, FACE_AVG), FACE_NAMES, FACE_NAMES_EN, "provisional_ratio")
    lip_up, lip_low = float(m['lip_up']), float(m['lip_low'])
    ltw = (lip_up + lip_low) / max(float(m['mouth_w']), eps)
    lip_idx = int(ltw >= LIP_CUTS[0]) + int(ltw >= LIP_CUTS[1]) + int(ltw > LIP_CUTS[2]) + int(ltw > LIP_CUTS[3])
    ulr = lip_up / max(lip_low, eps)
    bal_idx = 0 if ulr < LIPBAL_CUTS[0] else (2 if ulr <= LIPBAL_CUTS[1] else 4)
    corner = float(m['mouth_tilt'])
    cor_idx = 0 if corner < CORNER_CUTS[0] else (2 if corner <= CORNER_CUTS[1] else 4)
    lip_thick_item = _fixed_item('lip_thick', lip_idx, ltw, None, "lip thickness/mouth width %.2f (평균 0.44, 임시)" % ltw, LIP_NAMES, LIP_NAMES_EN, "korean_lip_average")
    lip_bal_item = _fixed_item('lip_balance', bal_idx, ulr, None, "upper/lower %.2f (평균 0.64~0.90, 임시)" % ulr, LIPBAL_NAMES, LIPBAL_NAMES_EN, "korean_lip_average")
    corner_item = _fixed_item('mouth_corner', cor_idx, corner, None, "mouth tilt %+.1f deg (보통 2~9, 임시)" % corner, CORNER_NAMES, CORNER_NAMES_EN, "provisional_geometry")
    if float(m['lip_h']) - lip_up - lip_low > MOUTH_OPEN_GAP:            # 입이 벌어져 있으면 두께·입꼬리가 부정확
        for it_ in (lip_thick_item, lip_bal_item, corner_item):
            it_["reliability"] = "low"
            it_["caveat"] = "입이 벌어져 있어 부정확할 수 있어요. 입을 다물고 힘을 뺀 무표정으로 다시 찍어 보세요"
    nose_ratio = float(m['nose_len']) / max(float(m['cheek_w']), eps)
    nose_item = _fixed_item('nose_len', nose_level(nose_ratio), nose_ratio, None,
                            "nose length / face width %.2f (임시 기준 평균 0.35)" % nose_ratio, NOSE_NAMES, NOSE_NAMES_EN, "provisional_ratio")
    if ref is not None and ref.has('eye_tilt'):
        rel, ratio = _reliability(ref, 'eye_tilt', std, n)
        if rel:
            tilt_item["reliability"], tilt_item["noise_ratio"] = rel, ratio

    if use_ref:
        items = {
            'face_ratio': face_item,
            'eye_tilt': tilt_item,
            'eye_open': _ref_item(ref, 'eye_open', 'eye_aspect', m['eye_aspect'], 1.0 / max(m['eye_aspect'], eps),
                                  "h/w %.2f (p%d, med %.2f)", std, n, invert=True),
            'eye_gap': gap_item,
            'brow_tilt': brow_item,
            'nose_len': nose_item,
            'mouth_w': _ref_item(ref, 'mouth_w', 'mouth_w', m['mouth_w'], m['mouth_w'], "mouth/IPD %.3f (p%d, med %.3f)", std, n),
            'lip_thick': lip_thick_item,
            'lip_balance': lip_bal_item,
            'mouth_corner': corner_item,
            'philtrum': phil_item,
        }
        z_open = items['eye_open']['z']
        z_size = _z_from_pct(ref.percentile('eye_len', m['eye_len']))            # 눈 길이(÷동공 사이)가 길수록 큰 눈
        reference = {"kind": "table", "source": Path(ref.path).name, "n": min(s_["count"] for s_ in ref.stats.values()),
                     "levels": "quintile cuts p20/p40/p60/p80 (eye opening, mouth)",
                     "note": "same landmark model and IPD normalization as the live measurement; population of the table is not Korean-specific"}
    else:
        eye_w_gap = m['eye_len'] / max(m['eye_gap'], eps)
        zf = lambda v, r: (v - r[0]) / r[1]
        z_open, z_size = zf(eye_open, LABEL_REF['eye_open']), zf(eye_w_gap, LABEL_REF['eye_w_gap'])
        z_mouth = zf(m['mouth_w'], LABEL_REF['mouth_ipd'])
        lv = lambda z: sum(z >= c for c in Z_CUTS)
        items = {
            'face_ratio': face_item,
            'eye_tilt': tilt_item,
            'eye_open': _label_item('eye_open', lv(z_open), eye_open, z_open,
                                    "h/w %.2f (ref %.2f+-%.2f)" % (eye_open, *LABEL_REF['eye_open']), "korean_adult_mean_sd"),
            'eye_gap': gap_item,
            'brow_tilt': brow_item,
            'nose_len': nose_item,
            'mouth_w': _label_item('mouth_w', lv(z_mouth), m['mouth_w'], z_mouth,
                                   "mouth/IPD %.2f (ref %.2f+-%.2f)" % (m['mouth_w'], *LABEL_REF['mouth_ipd']), "korean_adult_mean_sd"),
            'lip_thick': lip_thick_item,
            'lip_balance': lip_bal_item,
            'mouth_corner': corner_item,
            'philtrum': phil_item,
        }
        reference = {"kind": "paper", "source": "Korean adult mean+-SD from the knowledge-base document",
                     "note": "eye opening / mouth use paper values (definitions differ from our landmark model); labels may be shifted. "
                             "Put face_reference*.csv next to the script."}

    items.update({k: _flex_item(k, m, ref) for k in FLEX})
    items['symmetry'] = _symmetry_item(m, ref)
    items = {k: items[k] for k in LABEL_ORDER if k in items}
    lid = eyelid.get("label") if isinstance(eyelid, dict) and eyelid.get("status") == "measured" else None
    ranking = eye_shape_match(z_tilt, z_open, z_size, lid)
    best = ranking[0]
    return {
        "status": "measured",
        "reference": reference,
        "rules": rules_report(m, ref),
        "luck": luck_report(items),
        "criteria": {"eye_tilt": "user table: <5.0 / 5.0-7.3 / 7.4-9.6 / 9.7-12.0 / >12.0 deg (mean 8.5)",
                     "eye_gap": "user table: gap/eye width <1.15 / 1.15-1.30 / 1.30-1.45 / 1.45-1.60 / >1.60 (mean 1.38)"},
        "items": items,
        "eye_shape": {
            "best": best["name"], "best_en": best["name_en"], "similarity": best["similarity"],
            "distinct": bool(best["similarity"] >= EYE_DISTINCT_MIN), "description": best["description"],
            "ranking": ranking[:3], "ranking_all": ranking,
            "compared": ["눈꼬리 방향", "눈 열림", "눈 크기(눈 길이)" if use_ref else "눈 크기(눈 너비/두 눈 사이)"] + (["쌍꺼풀"] if lid else []),
            "z": {"tilt": round(z_tilt, 2), "open": round(float(z_open), 2), "size": round(float(z_size), 2)},
            "not_measurable": EYE_NOT_MEASURABLE,
            "caveat": "비교에 쓴 조건만 반영한 대략적 닮음이며, 지식 베이스의 16가지 중 7가지만 후보",
        },
        "sex_reference": sex if not use_ref else "not used (reference table is not sex-specific)",
        "notice": "재미로 보는 관상. 속설이며 과학적 근거가 없고, 사람을 판단하는 용도로 쓰면 안 돼요.",
    }


# =============================================================================
#  [5] face_fortune - 촬영 · 검사 · 저장 · 화면
# =============================================================================

ROOT = Path(__file__).resolve().parent
CAPTURE_DIR = ROOT / "captures"

CAMERA_FALLBACK = (0, 1, 2)
TARGET_FRAMES = 30          # 이만큼 '좋은' 프레임을 모으면 평균을 낸다
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

    def info(self) -> dict[str, Any]:
        """저장 파일에 남길 점 모델 정보 (풀이 창에 보인다)"""
        return {"model": "WFLW 98pt + 가상 5점 (ours)", "eye_refine": bool(getattr(self.mesh.refiner, "ok", False))}


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
                 eyelid: dict[str, Any] | None = None, labels: dict[str, Any] | None = None,
                 landmark_info: dict[str, Any] | None = None) -> Path:
    destination = next_capture_folder(directory)
    height, width = camera_frame.shape[:2]

    quality = {
        "stable": True,
        "sample_count": len(coordinates),
        "average_cv": measurement.score,
        "frontal": True,
        "single_face": True,
        "gaze_verified": True,
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
        eyelid=eyelid if eyelid is not None else eyelid_unavailable("eyelid check not run"),                       # 쌍꺼풀/홑꺼풀
        face_labels=labels if labels is not None else {"status": "unavailable", "reason": "labels not computed"},   # 5단계 라벨 + 닮은 눈모양
        landmark=landmark_info or {},
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
        "eyelid": analysis["eyelid"], "face_labels": analysis["face_labels"],
    }

    if not cv2.imwrite(str(destination / "face_measurement.jpg"), camera_frame):
        raise OSError("Could not save face_measurement.jpg")
    if not cv2.imwrite(str(destination / "measurement_screen.jpg"), screen_image):
        raise OSError("Could not save measurement_screen.jpg")
    for name, data in (("vision_landmarks.json", vision), ("analysis_result.json", analysis), ("handoff.json", handoff)):
        (destination / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return destination


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
    if any(w in s for w in ("BLINK", "LOOK", "MOVE", "KEEP")): return C_AMBER
    if any(w in s for w in ("UNSTABLE", "REQUIRED", "FAILED")): return C_RED
    return C_MUTED


def status_korean(status: str) -> str:
    s = status.upper()
    table = (("ONE FACE REQUIRED", "얼굴 한 명만 보여 주세요"), ("KEEP FACE INSIDE", "얼굴을 화면 안에 두세요"),
             ("MOVE CLOSER", "조금 더 가까이"), ("LOOK STRAIGHT", "정면을 봐 주세요"), ("BLINK", "눈 깜빡임 - 잠시만"),
             ("LOOK AT THE CAMERA", "카메라 렌즈를 봐 주세요"), ("AUTO-SAVED", "저장 완료 · R 새로 스캔"),
             ("AUTO SAVING", "측정 완료 · 저장 중"), ("UNSTABLE", "흔들림 · R 다시 시도"), ("SAVE FAILED", "저장 실패"),
             ("COLLECTING", "측정 중"))
    return next((k for e, k in table if e in s), status)


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
    if ui_state["mode"] == "gwansang":
        LABEL_UI["open_request"] = True                  # [관상] 을 누르면 풀이 창도 연다 (메인 루프가 처리)


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
    if mode is not None and mode["id"] == "gwansang":
        draw_gwansang_panel(panel, y, h)
        return
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

    draw_text(panel, "R new scan   S save   1 / 2 mode   V result window   Q quit", SEG_X, fy + 26, 0.34, C_DIM, 1)

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


# =============================================================================
#  [6] result window - '관상 풀이' 창.  스캔이 끝나 저장되면 새 창으로 자동으로 뜬다 (AI 연결 없음)
# =============================================================================
#
#  analysis_result.json 의 측정값·라벨을 읽어 눈 유형 · 부위별 5단계 · 눈꼬리 각도표 · 두 눈 사이 표 · 속설 · 측정값 전체를 보여준다.
#  카메라 루프와 따로 도는 프로세스라서 창이 열려 있어도 촬영이 느려지지 않는다.   python face_fortune_labels2.py --view captures/폴더
#  창 키:  ↑ ↓ / 마우스 휠 / PageUp PageDown / Home End 스크롤,  Q 또는 ESC 로 닫기

if not _PIL_OK:                       # pillow 가 없으면 창 대신 글자 출력으로 대체된다
    Image = ImageDraw = ImageFont = None

import re
import threading
import urllib.request

# AI 해설 (Ollama 의 OpenAI 호환 주소).  환경변수 AI_BASE_URL / AI_MODEL, 옵션 --ai-url / --ai-model, 끄기 --no-ai
AI = {"url": os.environ.get("AI_BASE_URL", "http://localhost:11434/v1"), "model": os.environ.get("AI_MODEL", "qwen2.5:1.5b"),
      "timeout": float(os.environ.get("AI_TIMEOUT", "120"))}
# 모델별 설정: 아주 작은 모델은 문장을 짧게, 온도를 낮춰서 말이 엉키는 걸 줄인다
AI_PRESETS = {"qwen2.5:0.5b": {"sentences": "2~3문장", "temperature": 0.5, "max_tokens": 200, "frequency_penalty": 0.8},
              "default": {"sentences": "3~4문장", "temperature": 0.5, "max_tokens": 300, "frequency_penalty": 0.3}}


def _ai_preset():
    return AI_PRESETS.get(AI["model"], AI_PRESETS["default"])

RW_W, RW_H = 960, 780
RW_PAD = 28
RW_BG, RW_CARD, RW_CARD2, RW_BORDER = (16, 13, 23), (31, 27, 43), (42, 37, 58), (64, 54, 82)
RW_ACCENT, RW_TEXT, RW_MUTED, RW_GREEN, RW_RED = (255, 196, 0), (235, 235, 240), (154, 151, 171), (115, 240, 145), (255, 130, 130)
RW_LEVEL = [(100, 175, 255), (150, 205, 255), (215, 205, 205), (255, 195, 120), (255, 150, 80)]     # 낮음(파랑) → 높음(주황)
RW_COLORS = {"text": RW_TEXT, "muted": RW_MUTED, "accent": RW_ACCENT, "green": RW_GREEN, "red": RW_RED}

AI_KEYS = ['face_ratio', 'eye_tilt', 'eye_open', 'eye_gap', 'brow_tilt', 'nose_len', 'mouth_w', 'lip_thick', 'lip_balance', 'mouth_corner', 'philtrum']
RW_SHORT = {'forehead': '이마', 'cheek': '광대', 'symmetry': '좌우 균형', 'eye_size': '눈 크기', 'brow_len': '눈썹 길이', 'brow_eye_gap': '전택궁', 'glabella': '미간',
            'face_ratio': '얼굴형', 'eye_tilt': '눈꼬리', 'eye_open': '눈 열림', 'eye_gap': '두 눈 사이', 'brow_tilt': '눈썹', 'nose_len': '코 길이', 'mouth_w': '입 크기', 'lip_thick': '입술 두께', 'lip_balance': '윗·아랫입술', 'mouth_corner': '입꼬리', 'philtrum': '인중'}
RW_LID = {"double": "쌍꺼풀", "single": "홑꺼풀"}
RW_ANGLE_KEYS = {'jaw_angle', 'jaw_angle_l', 'jaw_angle_r', 'eye_tilt', 'brow_tilt', 'mouth_tilt'}

# 속설 (문서 '관상 정보 총정리' + 사용자가 준 해석).  운명을 단정하는 부분은 뺐고, 평균 쪽에도 한 줄씩 넣었다.  band: low(1~2단계) / avg(3) / high(4~5)
RW_NOTE = {
    ('forehead', 'low'): '이마가 좁으면 초년에 스스로 길을 내는 실전형이라, 늦게 피어 오래 가는 운이라고 봐요.',
    ('forehead', 'avg'): '이마 폭이 평균이라 윗사람 복과 초년운이 큰 굴곡 없이 무탈하다고 봐요.',
    ('forehead', 'high'): '이마가 넓으면 부모 · 윗사람 복과 초년운이 좋고 지혜로우며, 관록(직업 · 지위) 운이 좋다고 봐요.',
    ('cheek', 'low'): '광대가 좁으면 온화하고 앞에 나서기보다 뒤에서 돕는 쪽이라, 사람과 부딪힘이 적다고 봐요.',
    ('cheek', 'avg'): '광대 폭이 평균이라 추진력과 온화함이 균형 잡혀 무탈하다고 봐요.',
    ('cheek', 'high'): '광대가 발달하면 소신과 추진력, 권위가 있어 명예와 직업운이 좋다고 보며, 스스로 일어서는 자수성가형이라는 말도 있어요.',
    ('symmetry', 'low'): '좌우가 고르면 기운이 안정적이라 건강운과 대인관계가 순탄하다고 봐요.',
    ('symmetry', 'avg'): '좌우 균형이 평균이라 기운의 흐름이 무난하다고 봐요.',
    ('symmetry', 'high'): '좌우가 조금 치우치면 운에 기복이 있을 수 있다고 보지만, 생활 습관과 표정으로 바뀐다고 봐요.',
    ('eye_size', 'low'): '눈이 작으면 신중하고 실속을 챙기며 비밀을 잘 지킨다고 봐요.',
    ('eye_size', 'avg'): '눈 크기가 평균이라 감성과 이성이 균형 잡혀 무난하다고 봐요.',
    ('eye_size', 'high'): '눈이 크면 감수성과 표현력이 풍부하고 사람을 끄는 매력이 있어 애정운이 좋다고 봐요.',
    ('brow_len', 'low'): '눈썹이 짧으면 형제 · 친구 복이 약하다고 보아 스스로 인맥을 넓혀야 한다고 해요. 대신 한번 맺은 인연은 깊게 가요.',
    ('brow_len', 'avg'): '눈썹 길이가 평균이라 필요한 만큼의 인연이 곁에 있고 무탈하다고 봐요.',
    ('brow_len', 'high'): '눈썹이 눈보다 길면 형제 · 친구 복과 인복이 좋고 대인관계가 원만하다고 봐요.',
    ('brow_eye_gap', 'low'): '눈썹과 눈 사이(전택궁)가 좁으면 집 · 부동산 운은 스스로 일궈야 하고 성격이 급할 수 있다고 봐요.',
    ('brow_eye_gap', 'avg'): '전택궁이 평균이라 집과 터전의 운이 무탈하다고 봐요.',
    ('brow_eye_gap', 'high'): '전택궁이 넓으면 좋은 집에서 살고 부동산 · 재물 운이 좋으며 마음에 여유가 있다고 봐요.',
    ('glabella', 'low'): '미간(명궁)이 좁으면 생각이 많고 신중하며 예민해서, 걱정을 덜어내면 운이 트인다고 봐요.',
    ('glabella', 'avg'): '미간이 평균이라 운의 흐름이 큰 굴곡 없이 무탈하다고 봐요.',
    ('glabella', 'high'): '미간(명궁)이 넓고 평평하면 마음이 넓고 운의 흐름이 순탄하며 귀하게 된다고 봐요.',
    ('face_ratio', 'low'): '얼굴이 둥글면 수형으로 보아 지혜롭고 속마음을 잘 드러내지 않으며, 온순하고 재물 관리에 밝다고 봐요. (학파에 따라 금형으로 보기도 해요)',
    ('face_ratio', 'avg'): '가로세로 비율이 평균이라 오행이 한쪽으로 치우치지 않은 무난하고 균형 잡힌 얼굴형으로 봐요.',
    ('face_ratio', 'high'): '세로가 긴 얼굴은 목형으로 보아 착하고 부드럽고 진취적이며 한 가지를 파고드는 타입이라고 봐요. 쉽게 화를 낸다는 말도 있어요.',
    ('eye_tilt', 'low'): '눈꼬리가 덜 올라간 눈은 기운이 아래로 향해 부드럽고 선량해 보인다고 봐요.',
    ('eye_tilt', 'avg'): '눈꼬리 각도가 평균 범위라 기운이 한쪽으로 치우치지 않고 무난하며 안정적인 인상이라고 봐요.',
    ('eye_tilt', 'high'): '눈꼬리가 많이 올라가면 기운이 위로 향해 활력이 있고 두뇌 회전이 빠르다고 보지만, 날카로워 보일 수 있다고도 해요.',
    ('eye_open', 'low'): '눈이 가늘고 길쭉하면 차분하고 속이 깊어 보이며 총명한 인상이라고 봐요.',
    ('eye_open', 'avg'): '눈 모양이 균형 잡혀 있어 무난하고 편안한 인상이라고 봐요.',
    ('eye_open', 'high'): '눈이 크고 둥글면 정이 많고 인내심 있는 실속형 인상이라고 봐요.',
    ('eye_gap', 'low'): '두 눈 사이가 좁으면 감정이 예민하고 걱정이 많으며 눈앞의 이익을 먼저 본다고 봐요.',
    ('eye_gap', 'avg'): '두 눈 사이가 표준이라 마음 씀씀이와 판단이 균형 잡혀 무난하다고 봐요.',
    ('eye_gap', 'high'): '두 눈 사이가 멀면 마음이 넓고 먼 곳을 지향한다고 봐요.',
    ('nose_len', 'low'): '짧은 코는 자기 생각보다 남이 시키는 대로 움직이는 경향이 있어요. 이끄는 사람이 지혜로우면 득이 되지만 그렇지 않으면 낭패를 볼 수 있다고 봐요. '
                         '일을 심각하게 생각하지 않는 편이고, 포용력과 결단력이 있으면 사람을 이끌지만 반대로 게으름을 피울 수도 있어요.',
    ('nose_len', 'avg'): '코 길이가 얼굴에 알맞아 자존심과 유연함이 균형 잡혀 있고 무탈한 편이라고 봐요.',
    ('nose_len', 'high'): '얼굴에 비해 긴 코는 자존심이 강하고 남다른 인내심이 있다고 봐요. 다소 융통성이 부족하지만 책임감이 강해 맡은 일에 최선을 다해서 주변이 안심하는 타입이에요. '
                          '예민해서 신경질적인 태도는 주의하라고 해요.',
    ('mouth_w', 'low'): '입이 작으면 욕심이 적고 남을 돕기를 좋아하며 예리한 식견이 있다고 봐요. 다만 소심하고 끈기가 부족할 수 있다고도 해요.',
    ('mouth_w', 'avg'): '입 크기가 평균이라 욕심과 소심함이 한쪽으로 치우치지 않고 무탈한 편이라고 봐요.',
    ('mouth_w', 'high'): '입이 크면 욕심과 생활력이 왕성하고 담력이 커서 리더 기질이 있다고 봐요.',
    ('lip_thick', 'low'): '입술이 얇으면 유머와 화술이 뛰어나 말을 잘하고 이성적이지만, 냉정해 보이거나 정이 박한 말을 할 수 있다고 봐요. 재물을 담는 그릇이 얕다고 보아 지키는 노력이 필요하다고 하지만, 자주 웃고 긍정적으로 말하면 바뀐다고도 해요.',
    ('lip_thick', 'avg'): '입술 두께가 평균이라 정과 이성이 균형 잡혀 있고, 애정운과 인복이 무난하고 무탈하다고 봐요.',
    ('lip_thick', 'high'): '도톰한 입술은 정이 많고 감수성이 풍부하며 봉사적이라고 봐요. 풍요로운 애정운과 인복, 안정적인 살림운으로 연결해요.',
    ('lip_balance', 'low'): '아랫입술은 욕(欲)과 받는 사랑으로 봐서, 욕구에 충실하고 상대가 챙겨 주길 바라는 면이 있지만 건강과 정력이 좋다고 봐요. 아랫입술이 도톰하면 말년의 재물운이 안정적이라는 해석도 있어요.',
    ('lip_balance', 'avg'): '위아래 입술이 균형 잡혀 있으면 표현력, 사회성, 연애 능력이 좋고 결혼운이 길고 순탄하다고 봐요.',
    ('lip_balance', 'high'): '윗입술은 정(情)과 주는 사랑으로 봐서, 감수성과 언어 능력이 뛰어나고 가정적이지만 감정 절제가 약하고 오지랖이 넓을 수 있다고 봐요.',
    ('mouth_corner', 'low'): '입꼬리가 처지면 마음이 무겁고 걱정이 많아 보인다고 하고, 말년의 흐름이 힘들 수 있다는 해석이 있어요. 다만 자주 웃으면 바뀐다고 봐요.',
    ('mouth_corner', 'avg'): '입꼬리가 평평해 감정이 차분하고 안정적이며, 말년의 흐름도 무탈하다고 봐요.',
    ('mouth_corner', 'high'): '입꼬리가 올라가면 긍정적이고 사교성이 좋아 인기(도화)와 배우자복이 있고 말년이 행복하다고 봐요.',
    ('philtrum', 'low'): '인중이 짧으면 고집이 약하거나 결정력이 부족하다고 보고, 이성 관계에서 갈등이 있을 수 있다고 해요.',
    ('philtrum', 'avg'): '인중이 평균 길이라 생명력과 자식운, 말년의 흐름이 큰 굴곡 없이 무탈하다고 봐요.',
    ('philtrum', 'high'): '인중이 길면 배우자와의 관계가 원만할 가능성이 크고 자녀복이 있다고 봐요. 인내심이 깊고 생각이 바르지만 사람을 잘 믿어 손해를 볼 수 있다고도 해요.',
}
RW_BROW_NOTE = {   # 눈썹 각도는 4구간이라 단계(1·2·3·5)별로
    1: '아래로 내려간 팔자 눈썹은 느긋하고 얌전해 보이지만 남에게 기대지 않고 스스로 길을 개척하는 타입이라고 봐요. 가끔 예상 못 한 실수로 주변 도움을 받지만, 배려심 많고 상냥하며 자기 페이스대로 움직여요.',
    2: '일자형 눈썹은 결단력이 있어 한번 정한 일은 끝까지 해낸다고 봐요. 거짓말을 잘 못하고 자기중심적이며 겉과 속이 같은 솔직한 타입이고, 자신만의 규칙을 엄격히 지켜요.',
    3: '눈썹 각도가 평균 범위라 성격이 무난하고 균형 잡힌 편이라고 봐요. 한쪽으로 크게 치우치지 않아 사람들과 두루 잘 지내요.',
    5: '위로 솟아오른 눈썹은 욕심이 많고 성격이 급한 편이라고 봐요. 예술 쪽 재능을 보이기도 하고, 맡은 일은 확실히 열심히 해요. 예민해서 다소 신경질적일 수 있고 남의 평가에 민감해요.',
}
# 운세 7가지의 보충법.  오행 배속(색 · 방향 · 맛 · 숫자 · 물건)과 풍수 속설에서 모았다
RW_ELEMENTS = {
    '木': ('목', '초록 · 청록', '동쪽', '신맛 · 푸른 채소', '3 · 8', '살아 있는 식물, 나무 소품'),
    '火': ('화', '빨강 · 분홍 · 보라', '남쪽', '쓴맛 (차 조금)', '2 · 7', '은은한 조명, 초, 햇빛 드는 자리'),
    '土': ('토', '노랑 · 갈색 · 베이지', '가운데 · 남서쪽', '단맛 · 곡물 · 뿌리채소', '5 · 10', '도자기, 흙 화분, 둥근 소품'),
    '金': ('금', '흰색 · 은색 · 금색', '서쪽', '매운맛 · 흰 음식', '4 · 9', '금속 소품, 동전, 시계'),
    '水': ('수', '검정 · 남색', '북쪽', '짠맛 · 해산물 · 검은콩', '1 · 6', '깨끗한 물의 수경 식물, 유리 소품'),
}
RW_LUCK = {
    'wealth': (['金', '土'], '재물을 쌓는 금(金)과 지키는 토(土) 기운이 약하다고 봐요. 버는 것보다 새는 곳을 막는 쪽이 먼저예요.',
               '벌고 쓰는 균형이 잡힌 편이에요. 습관 하나로 결과가 갈린다고 봐요.', '재물이 들어오면 머무는 상이라고 봐요. 지키는 습관만 이어 가면 돼요.',
               ['황금색 · 노란색 · 갈색 장지갑 (돈이 접히지 않게)', '지폐를 한 방향으로 정리한 지갑', '현관에 금전수나 스투키', '동전 · 금속 소품'],
               ['빨간 지갑 (불이 금을 녹인다는 속설)', '영수증이 가득하거나 해진 지갑', '현관에 쌓인 신발과 잡동사니', '보증 · 한 번에 만회하려는 투자']),
    'love': (['火', '木'], '인연을 끌어오는 화(火, 도화)와 생기를 주는 목(木) 기운이 약하다고 봐요. 표현을 늘리면 채워진다고 해요.',
             '천천히 깊어지는 인연 쪽이에요.', '매력이 먼저 드러나 인연이 먼저 다가오는 상이라고 봐요.',
             ['분홍색 소품', '살아 있는 꽃 (장미 · 작약 · 복숭아꽃)', '짝수로 둔 소품과 화분', '하트 잎 식물 (하트호야)', '남서쪽(인연 방향) 정리'],
             ['침대를 정면으로 비추는 거울', '시든 꽃 · 먼지 쌓인 조화', '어수선한 침실과 머리맡 잡동사니', '한쪽만 벽에 붙인 침대 (배우자 자리가 막힌다는 속설)']),
    'health': (['水', '木'], '회복을 돕는 수(水)와 생기를 주는 목(木) 기운이 약하다고 봐요. 몰아서 쓰지 말고 고르게 쉬는 것이 보충법이에요.',
               '규칙적이면 큰 탈이 없는 편이에요.', '타고난 체력이 좋고 회복이 빠른 상이라고 봐요. 쉬는 것도 계획에 넣으세요.',
               ['초록 식물 (행운목)', '환기가 잘 되는 침실과 정해진 수면 시간', '따뜻하고 물기 없는 욕실', '검정 · 남색 침구'],
               ['고인 물 · 오래된 꽃병 물 (수독)', '침실 거울 · 머리맡 충전기와 잡동사니', '밤샘과 폭식']),
    'career': (['木', '火'], '자라나는 목(木)과 이름을 알리는 화(火) 기운이 약하다고 봐요. 기술 하나를 꾸준히 쌓는 것이 보충법이에요.',
               '꾸준함이 곧 경력이 되는 편이에요.', '자리를 얻고 밀고 나가는 힘이 있는 상이라고 봐요.',
               ['책상 옆 행운목 (승진 · 학업)', '정돈된 책상', '등 뒤가 벽인 자리', '빨강 · 초록 포인트 소품'],
               ['어지러운 책상과 서류 더미', '등 뒤가 문이나 창인 자리', '고장 난 시계와 물건 방치']),
    'people': (['土', '木'], '사람을 품는 토(土) 기운이 약하다고 봐요. 먼저 안부를 묻는 습관이 보충법이에요.',
               '필요한 사람이 곁에 남는 편이에요.', '사람이 모이고 말이 통하는 상이라고 봐요. 거절하는 법도 익혀 두세요.',
               ['정리된 현관 (관계운은 입구에서 시작된다는 말)', '노랑 · 베이지 같은 따뜻한 색', '둥근 소품과 식탁 위 꽃', '웃는 입꼬리'],
               ['사람이 머무는 곳을 비추는 마주 보는 거울', '현관의 날카로운 장식 · 가시 식물', '말하지 않고 쌓아 두는 서운함']),
    'family': (['土'], '터전을 받치는 토(土) 기운이 약하다고 봐요. 집을 정돈하는 것부터가 보충법이에요.',
               '집과 가족의 흐름이 무난한 편이에요.', '집과 터전, 자녀 복이 든든한 상이라고 봐요.',
               ['가족사진 · 풍경화', '뿌리를 내린다는 분재, 거실의 행운목이나 아이비', '갈색 · 노랑 · 베이지 소품', '함께 밥 먹는 식탁'],
               ['깨진 그릇 · 금 간 컵', '막힌 배수구 · 정체된 물', '현관과 거실의 잡동사니']),
    'late': (['水', '土'], '오래 쌓는 수(水)와 받쳐 주는 토(土) 기운이 약하다고 봐요. 젊을 때 기반을 만들어 두면 편안하다고 해요.',
             '말년이 큰 굴곡 없이 흘러가는 편이에요.', '나이 들수록 사람이 모이고 편안해지는 상이라고 봐요.',
             ['자동이체 저축 같은 꾸준한 습관', '검정 · 남색 소품', '북쪽 공간 정돈', '자주 웃기 (입꼬리가 말년을 연다는 말)'],
             ['무리한 빚과 보증', '처진 표정 습관', '한 번에 크게 바꾸려는 결정']),
}
RW_FACE_HEAD = '가로가 넓고 각지면 금형, 광대가 발달한 마름모는 토형, 위가 좁고 아래가 넓으면 화형으로 보는 해석도 있어요. 이 비율만으로는 가려내지 못해요.'
RW_PHILTRUM_HEAD = '인중은 생명력, 자식운, 말년의 흐름과 연결해서 봐요.'


def _rw_band(level):
    return 'low' if level <= 2 else ('high' if level >= 4 else 'avg')


def _rw_note(key, level):
    """(속설 문장, 해석 자료가 있는지)"""
    if key == 'brow_tilt':
        return RW_BROW_NOTE.get(level, ''), True
    t = RW_NOTE.get((key, _rw_band(level)))
    return (t, True) if t else ('이 쪽에 대한 속설 자료는 아직 없어요.', False)


# ---- 읽을거리 만들기: analysis_result.json → 창과 reading.txt 가 같이 쓰는 구조 ----------------------------------

def _rw_fmt_metric(name, v):
    if v is None:
        return '--'
    if name in RW_ANGLE_KEYS:
        return '%.1f°' % v
    if name.startswith('zone_'):
        return '%.1f%%' % v
    return '%.3f' % v


def luck_section(labels):
    """운세 7가지: 상태 · 근거 · 도움 되는 기운(오행 색 · 방향 · 맛 · 숫자 · 물건) · 지니면 좋은 것 · 피할 것"""
    blocks = [{'t': 'p', 'text': '얼굴 부위와 운의 연결(십이궁 등)로 계산한 재미용 풀이예요. 부족하면 도움 되는 기운과 물건을, 좋으면 지키는 법을 같이 적었어요.',
               'size': 14, 'color': 'muted'}]
    for lk in labels.get('luck') or []:
        els, lack, mid, high, carry, avoid = RW_LUCK[lk['key']]
        word = {'low': '부족해요', 'mid': '보통이에요', 'high': '좋아요'}[lk['state']]
        blocks.append({'t': 'p', 'text': '%s — %s' % (lk['name'], word), 'size': 19, 'bold': True, 'brief': True, 'after': 2,
                       'color': {'low': 'red', 'mid': 'text', 'high': 'green'}[lk['state']]})
        blocks.append({'t': 'p', 'text': {'low': lack, 'mid': mid, 'high': high}[lk['state']], 'size': 16, 'brief': True, 'after': 2})
        why = ('올려 준 것: ' + ', '.join(lk['ups']) if lk['ups'] else '') + ('   끌어내린 것: ' + ', '.join(lk['downs']) if lk['downs'] else '')
        if why:
            blocks.append({'t': 'p', 'text': why.strip(), 'size': 13, 'color': 'muted', 'after': 2})
        for e in els:
            ko, color, direction, taste, number, things = RW_ELEMENTS[e]
            blocks.append({'t': 'p', 'text': '%s 기운 %s(%s): 색 %s / 방향 %s / 맛 %s / 숫자 %s / 물건 %s' % (
                '채울' if lk['state'] == 'low' else '도움 되는', e, ko, color, direction, taste, number, things), 'size': 15, 'brief': lk['state'] == 'low', 'after': 2})
        blocks.append({'t': 'p', 'text': ('지니면 좋은 것: ' if lk['state'] == 'low' else '지키려면: ') + ' · '.join(carry), 'size': 15, 'brief': True, 'after': 2})
        blocks.append({'t': 'p', 'text': '피할 것: ' + ' · '.join(avoid), 'size': 15, 'brief': True, 'after': 12})
    blocks.append({'t': 'p', 'text': '※ 건강은 관상보다 병원이 먼저예요. 몸이 불편하면 진료를 받아 주세요.', 'size': 13, 'color': 'muted'})
    return ('운세 7가지 · 부족한 기운 · 지닐 것 · 피할 것', blocks)


def build_reading(analysis, vision=None, folder_name='', ref=None, sex='any'):
    """분석 결과 dict → 읽을거리 dict. 라벨이 JSON 에 없으면 측정값(feature_values_ipd)에서 다시 계산한다. 측정값도 없으면 None"""
    labels = analysis.get('face_labels')
    recomputed = False
    if not labels or labels.get('status') != 'measured' or 'eye_shape' not in labels:
        ipd = analysis.get('feature_values_ipd')
        if not ipd:
            return None
        labels = face_labels(ipd, analysis.get('eyelid'), sex, ref, analysis.get('feature_std_ipd'),
                             (analysis.get('quality') or {}).get('sample_count'), eye_tilts(analysis.get('eyes_wflw')))
        recomputed = True
    items, es = labels['items'], labels['eye_shape']
    lid = analysis.get('eyelid') or {}
    top = es['ranking'][0]
    ranking = es.get('ranking_all') or es['ranking']
    second = es['ranking'][1] if len(es['ranking']) > 1 else None

    # ---- 맨 위 카드 ----
    head = top['name']                                         # 항상 가장 가까운 눈 하나를 보여준다
    if es['distinct']:
        note = ('  ·  %s도 가까워요' % second['name']) if second and second['similarity'] >= 0.2 else ''
    else:
        note = '  ·  평균에 가까워서 닮은 정도는 낮아요 (%d%%)' % int(top['similarity'] * 100)
    chips = []
    for key in LABEL_ORDER:                                    # 칩은 보통이 아닌 것만 (보통은 개수로)
        it = items[key]
        if it['level'] != 3:
            chips.append(('%s %s%s' % (RW_SHORT[key], it['label'], ' ~' if it.get('reliability') == 'low' else ''), True))
    n_avg = sum(1 for k in LABEL_ORDER if items[k]['level'] == 3)
    if n_avg:
        chips.append(('보통 %d개' % n_avg, False))
    if lid.get('status') == 'measured':
        chips.append(('%s %d%%' % (RW_LID.get(lid['label'], lid['label']), int(lid['probabilities'][lid['label']] * 100)), True))

    S = []                                                   # 본문 구역들
    # ---- A. 기본 해설 ----
    a = []
    if es['distinct']:
        t = '눈: %s에 가까워요 (닮은 정도 %d%%). %s.' % (top['name'], int(top['similarity'] * 100), top['description'])
        if second and second['similarity'] >= 0.2:
            t += ' %s의 느낌도 함께 있어요 (%d%%).' % (second['name'], int(second['similarity'] * 100))
    else:
        t = '눈: 평균에 가까운 눈이지만 굳이 고르면 %s 쪽이에요 (닮은 정도 %d%%, 낮아요). %s.' % (top['name'], int(top['similarity'] * 100), top['description'])
    a.append({'t': 'p', 'text': t})
    for key in LABEL_ORDER:
        it = items[key]
        txt, _ = _rw_note(key, it['level'])
        a.append({'t': 'p', 'text': '%s · %s%s: %s' % (it['name_ko'], it['label'], ' (경계)' if it.get('reliability') == 'low' else '', txt), 'after': 2})
    if lid.get('status') == 'measured':
        a.append({'t': 'p', 'text': '눈꺼풀: %s 쪽이에요 (%d%%).' % (RW_LID.get(lid['label'], lid['label']), int(lid['probabilities'][lid['label']] * 100)), 'after': 2})
    S.append(('한눈에 보기 (기본 해설)', a))
    rules = labels.get('rules') or {}
    virtual_top = (analysis.get('hairline') or {}).get('status') != 'hairline_segmentation'
    back = []                                                  # 쌍꺼풀 뒤에 붙일 구역 (삼정 · 십이궁 · 특징)
    if rules.get('status') in ('ok', 'no_reference'):
        oh, gr = rules['ohaeng'], rules.get('grade')
        chips.insert(0, ('%s(%s)' % (oh['name'], oh['hanja']), False))
        if gr:
            chips.insert(1, (gr['grade'].split(' · ')[0], gr['grade'][:1] in 'SA'))
            S.append(('종합 등급', [{'t': 'p', 'text': gr['grade'], 'size': 24, 'bold': True, 'color': 'accent', 'brief': True},
                                {'t': 'p', 'text': gr['one'], 'brief': True},
                                {'t': 'p', 'text': '십이궁 중 가장 든든한 단계인 궁이 %d / %d개예요.' % (gr['strong'], gr['total']), 'size': 14, 'color': 'muted'}]))
        S.append(('얼굴형 (오행)', [
            {'t': 'p', 'text': '%s %s · %s' % (oh['name'], oh['hanja'], oh['tag']), 'size': 22, 'bold': True, 'color': 'accent', 'brief': True},
            {'t': 'p', 'text': oh['one'], 'brief': True},
            {'t': 'p', 'text': oh['desc']},
            {'t': 'p', 'text': '강점: ' + ' · '.join(oh['strengths']), 'size': 16},
            {'t': 'p', 'text': '주의: ' + ' · '.join(oh['cautions']), 'size': 16},
            {'t': 'p', 'text': '어울리는 일: ' + oh['job'], 'size': 16}]))
        S.append(luck_section(labels))
        if rules.get('fortunes'):
            fb = []
            for f in rules['fortunes']:
                fb.append({'t': 'levelrow', 'name': f['name'], 'label': f['title'], 'level': [1, 3, 5][f['level'] - 1], 'caption': f['parts'],
                           'extra': [f['body'], '주의: ' + f['caution'], '조언: ' + f['advice']], 'caveat': None})
            S.append(('운세 5가지 (재물 · 직업 · 건강 · 인간관계 · 애정)', fb))
        else:
            S.append(('운세 · 십이궁', [{'t': 'p', 'text': rules.get('reason', '기준표(face_reference*.csv)가 있어야 운세 · 십이궁 · 등급이 계산돼요.'), 'color': 'muted', 'brief': True}]))
        zb = []
        for z in rules['zones']:
            zb.append({'t': 'p', 'text': '%s · %s  —  %s (%.1f%%)' % (z['name'], z['period'], {'short': '짧은 편', 'long': '긴 편', 'even': '균형'}[z['state']], z['value']),
                       'size': 17, 'bold': True, 'after': 2})
            zb.append({'t': 'p', 'text': z['text'], 'size': 16})
        if virtual_top:
            zb.append({'t': 'p', 'text': '※ 이마 위가 가상 점(헤어라인 분할 모델 없음)이라 위·가운데 구역이 같게 나와요. 초년운은 참고만 해 주세요.', 'size': 13, 'color': 'muted'})
        back.append(('삼정 (초년 · 중년 · 말년)', zb))
        if rules.get('palaces'):
            back.append(('십이궁', [{'t': 'levelrow', 'name': '%s %s' % (p['name'], p['hanja']), 'label': p['badge'], 'level': [1, 2, 4, 5][p['badge_idx']],
                                   'caption': p['region'], 'extra': [p['text']], 'caveat': None} for p in rules['palaces']]))
        if 'traits' in rules:
            tb = [{'t': 'p', 'text': '%s (상위·하위 %d%%): %s' % (t['name'], int(round(min(t['percentile'], 100 - t['percentile']))) , t['text']), 'size': 16}
                  for t in rules['traits']] or [{'t': 'p', 'text': '평균에서 크게 벗어난 항목이 없어요.', 'color': 'muted'}]
            back.append(('눈에 띄는 특징 (평균에서 많이 벗어난 항목)', tb))
    else:
        S.append(luck_section(labels))
        S.append(('운세 · 십이궁', [{'t': 'p', 'text': rules.get('reason', '규칙집(gwansang_rules.json)이 없어서 운세 · 십이궁은 건너뛰었어요.'), 'color': 'muted', 'brief': True}]))

    # ---- B. 눈 모양 닮은 정도 ----
    b = [{'t': 'bars', 'rows': [(r['name'], r['similarity'], '%d%%' % int(r['similarity'] * 100)) for r in ranking]},
         {'t': 'p', 'text': '닮은 정도는 100%%에 가까울수록 그 눈의 기준 모양과 비슷하다는 뜻이고, %d%% 아래면 뚜렷한 유형이 없다고 봐요.' % int(EYE_DISTINCT_MIN * 100),
          'size': 15, 'color': 'muted'}]
    for r in ranking[:3]:
        b.append({'t': 'p', 'text': '%s: %s' % (r['name'], r['description']), 'size': 16, 'after': 2})
    b.append({'t': 'kv', 'cols': 3, 'rows': [('눈꼬리 z', '%+.2f' % es['z']['tilt']), ('눈 열림 z', '%+.2f' % es['z']['open']), ('눈 크기 z', '%+.2f' % es['z']['size'])]})
    b.append({'t': 'p', 'text': 'z 점수는 0이 평균이고, 눈꼬리·눈 열림·눈 크기를 한국인 평균에 견줘 계산한 값이에요. 비교에 쓴 조건: %s.' % ' · '.join(es['compared']),
              'size': 15, 'color': 'muted'})
    b.append({'t': 'p', 'text': '못 재서 뺀 조건: %s. 그래서 이리눈·사백안·뱀눈·돼지눈 등은 후보에서 빠져 있어요.' % ' · '.join(es['not_measurable']),
              'size': 15, 'color': 'muted'})
    S.append(('눈 모양 닮은 정도', b))

    # ---- C. 눈꼬리 각도 ----
    it = items['eye_tilt']
    ew = analysis.get('eyes_wflw') or {}
    c = [{'t': 'p', 'text': '눈 안쪽 눈초리에서 바깥 눈초리로 이은 선의 기울기예요. 바깥이 더 높을수록 +이고, 왼쪽·오른쪽 눈 중 더 큰 값으로 판정해요.',
          'size': 16, 'color': 'muted'},
         {'t': 'scale', 'rows': TILT_SCALE, 'current': it['level'], 'value_text': '지금 값  %+.1f°  →  %d단계 (%s)' % (it['value'], it['level'], it['label'])}]
    rows = [('판정에 쓴 값 (큰 쪽 눈)', '%+.1f°' % it['value']), ('한국인 평균과 차이', '%+.1f°' % (it['value'] - TILT_AVG))]
    if ew.get('left') and ew.get('right'):
        tl, tr = ew['left']['metrics']['outer_corner_tilt_deg'], ew['right']['metrics']['outer_corner_tilt_deg']
        rows += [('사진 왼쪽 눈', '%+.1f°' % tl), ('사진 오른쪽 눈', '%+.1f°' % tr), ('좌우 차이', '%.1f°' % abs(tl - tr))]
    c.append({'t': 'kv', 'cols': 2, 'rows': rows})
    txt, _ = _rw_note('eye_tilt', it['level'])
    c.append({'t': 'p', 'text': '속설: ' + txt, 'size': 16})
    if it.get('caveat'):
        c.append({'t': 'p', 'text': '※ ' + it['caveat'], 'size': 14, 'color': 'muted'})
    S.append(('눈꼬리 각도 (5단계)', c))

    # ---- D. 두 눈 사이 거리 ----
    it = items['eye_gap']
    m = analysis.get('feature_values_ipd') or {}
    d_ = [{'t': 'p', 'text': '비율 = 두 눈 사이 거리(안쪽 눈초리 사이) ÷ 눈 가로 길이. 한국인 20~30대 평균 1.38을 표준(3단계) 중심으로 나눴어요.',
           'size': 16, 'color': 'muted'},
          {'t': 'scale', 'rows': GAP_SCALE, 'current': it['level'], 'value_text': '지금 값  %.2f  →  %d단계 (%s)' % (it['value'], it['level'], it['label'])}]
    rows = [('두 눈 사이 ÷ 눈 길이', '%.2f' % it['value']), ('한국인 평균과 차이', '%+.2f' % (it['value'] - GAP_AVG))]
    if m.get('eye_gap') is not None:
        rows += [('두 눈 사이 ÷ 동공 거리', '%.3f' % m['eye_gap']), ('눈 길이 ÷ 동공 거리', '%.3f' % m['eye_len'])]
    if m.get('eye_len_l') is not None:
        rows += [('왼쪽 눈 길이 (사진)', '%.3f' % m['eye_len_l']), ('오른쪽 눈 길이 (사진)', '%.3f' % m['eye_len_r'])]
    d_.append({'t': 'kv', 'cols': 2, 'rows': rows})
    txt, _ = _rw_note('eye_gap', it['level'])
    d_.append({'t': 'p', 'text': '속설: ' + txt, 'size': 16})
    if it.get('caveat'):
        d_.append({'t': 'p', 'text': '※ ' + it['caveat'], 'size': 14, 'color': 'muted'})
    S.append(('두 눈 사이 거리 (5단계)', d_))

    # ---- E. 그 밖의 부위 ----
    e = []
    for key in [k for k in LABEL_ORDER if k not in ('eye_tilt', 'eye_gap')]:
        it = items[key]
        extra = [RW_PHILTRUM_HEAD] if key == 'philtrum' else []
        txt, _ = _rw_note(key, it['level'])
        extra.append(txt)
        if key == 'face_ratio':
            extra.append(RW_FACE_HEAD)
        if it.get('percentile') is not None:
            extra.append('기준표에서 하위 %d%% 위치예요.' % int(round(it['percentile'])))
        if it.get('reliability') == 'low':
            extra.append('사람 사이 차이가 작은 항목이라 라벨이 한 칸 흔들릴 수 있어요.')
        e.append({'t': 'levelrow', 'name': it['name_ko'], 'label': it['label'], 'level': it['level'], 'caption': it['caption'],
                  'extra': extra, 'caveat': it.get('caveat')})
    S.append(('그 밖의 부위 (전체)', e))

    # ---- F. 쌍꺼풀 ----
    f = []
    if lid.get('status') == 'measured':
        f.append({'t': 'bars', 'rows': [(RW_LID.get(k, k), v, '%d%%' % int(v * 100)) for k, v in lid['probabilities'].items()]})
        rows = [('판정', RW_LID.get(lid['label'], lid['label'])), ('판정에 쓴 프레임', '%d개' % lid['n_samples']),
                ('눈 너비', '%.0f px%s' % (lid.get('eye_width_px', 0), '  (작음)' if lid.get('low_resolution') else ''))]
        if lid.get('model_test_accuracy') is not None:
            rows.append(('모델 시험 정확도', '%.1f%%' % (100 * float(lid['model_test_accuracy']))))
        f.append({'t': 'kv', 'cols': 2, 'rows': rows})
        f.append({'t': 'p', 'text': '이 카메라에서 정확도를 확인한 값이 아니라서 참고만 해 주세요.', 'size': 14, 'color': 'muted'})
    else:
        f.append({'t': 'p', 'text': '쌍꺼풀 판정은 꺼져 있어요: %s' % lid.get('reason', '모델 없음'), 'color': 'muted'})
        f.append({'t': 'p', 'text': 'eyelid.tflite 와 meta.json 을 이 파일 옆에 두면 켜져요. 켜지면 코끼리눈·소눈 닮은 정도에 반영돼요.', 'size': 14, 'color': 'muted'})
    S.append(('쌍꺼풀 / 홑꺼풀', f))
    S += back

    # ---- G. 측정값 전체 ----
    g = [{'t': 'p', 'text': '눈동자 사이 거리를 1로 놓은 비율이에요 (각도는 °, 구역은 %). ± 는 30프레임 사이의 흔들림이에요.', 'size': 15, 'color': 'muted'}]
    ko, sd = METRIC_KO, analysis.get('feature_std_ipd') or {}
    for gname, keys in METRIC_GROUPS:
        rows = [[ko[k], _rw_fmt_metric(k, m[k]), ('±' + _rw_fmt_metric(k, sd[k]).rstrip('°%')) if k in sd else ''] for k in keys if k in m]
        if rows:
            g.append({'t': 'table', 'title': gname, 'head': ['항목', '값', '흔들림'], 'rows': rows, 'widths': [0.52, 0.26, 0.22], 'aligns': 'lrr'})
    S.append(('측정값 전체 (비율 %d개)' % len(m), g))

    # ---- H. 자세 · 품질 · 측정 정보 ----
    pose, q, hl = analysis.get('pose') or {}, analysis.get('quality') or {}, analysis.get('hairline') or {}
    lm = analysis.get('landmark') or {}
    h = []
    rows = []
    if pose:
        rows += [('좌우 돌림 (yaw)', '%+.3f' % pose.get('yaw_ratio', 0)), ('고개 기울기 (roll)', '%+.1f°' % pose.get('roll_deg', 0)),
                 ('위아래 (pitch 근사)', '%.3f' % pose.get('pitch_proxy', 0))]
    rows += [('모은 프레임', '%s개' % q.get('sample_count', '?')), ('흔들림 평균 (CV)', '%.4f' % q['average_cv'] if q.get('average_cv') is not None else '--'),
             ('안정 판정', '안정' if q.get('stable') else '불안정')]
    h.append({'t': 'kv', 'cols': 2, 'rows': rows})
    rows = []
    if lm:
        rows += [('점 모델', lm.get('model', 'WFLW 98점')), ('눈 정밀 모델', '사용' if lm.get('eye_refine') else '안 씀 (1단계 점 그대로)')]
    rows += [('이마 상단', {'hairline_segmentation': '헤어라인 분할', 'virtual_point': '가상 점'}.get(hl.get('status'), str(hl.get('status', '--'))))]
    if vision:
        rows += [('사진 크기', '%s × %s' % (vision.get('image_width', '?'), vision.get('image_height', '?'))),
                 ('촬영 시각', str(vision.get('captured_at_utc', ''))[:19].replace('T', ' ') + ' UTC')]
    if folder_name:
        rows.append(('캡처 폴더', folder_name))
    h.append({'t': 'kv', 'cols': 1, 'rows': rows})
    if q.get('checks'):
        h.append({'t': 'p', 'text': '검사 조건: ' + q['checks'], 'size': 13, 'color': 'muted'})
    S.append(('자세 · 품질 · 측정 정보', h))

    # ---- I. 기준과 안내 ----
    ref_ = labels.get('reference', {})
    i = [{'t': 'p', 'text': '눈꼬리 각도와 두 눈 사이 거리는 사용자가 정한 고정 표로 5단계를 나눴어요 (각 구역 표 참고).', 'size': 16},
         {'t': 'p', 'text': '눈 열림·입은 %s로 20%%씩 5단계를 나눠요.' % (
             '기준표 %s (n=%s)' % (ref_.get('source', '?'), ref_.get('n', '?')) if ref_.get('kind') == 'table' else '문서의 한국 성인 논문 평균±편차'), 'size': 16},
         {'t': 'p', 'text': '인중은 코밑~윗입술 : 아랫입술 아래~턱끝 = 1:2 를 이상으로 두고, 1:1 에 가까울수록 긴 인중으로 봐요.', 'size': 16}]
    if recomputed:
        i.append({'t': 'p', 'text': '저장된 라벨이 없어 측정값으로 다시 계산했어요.', 'size': 14, 'color': 'muted'})
    i.append({'t': 'p', 'text': '눈썹 각도는 0° 미만 내려감(팔자) / 0~5° 일자형 / 5~12° 평균 / 12° 이상 위로 솟음으로 나눠요. 코 길이는 코 길이 ÷ 광대 폭이고 기준은 임시값이에요.', 'size': 16})
    if rules.get('status') in ('ok', 'no_reference'):
        i.append({'t': 'p', 'text': '오행 · 삼정 · 십이궁 · 운세 · 등급은 규칙집(gwansang_rules.json)을 바탕으로 한 재미용 풀이예요. 캡처 사진은 captures 폴더에 저장돼요.', 'size': 16})
    i.append({'t': 'p', 'text': labels.get('notice', '재미로 보는 관상이에요.'), 'color': 'green'})
    S.append(('기준과 안내', i))

    return {'labels': labels, 'eyelid': lid, 'title': '관상 풀이', 'head': head, 'note': note, 'chips': chips, 'sections': S,
            'provisional': ref_.get('kind') != 'table',
            'footer': ['※ 재미로 보는 관상이에요. 과학적으로 검증된 분석이 아니에요.',
                       '※ 눈꼬리·두 눈 사이·눈썹은 사용자 기준표, 코 길이는 임시 비율, 나머지는 %s 기준이에요.' % ('기준표 CSV' if ref_.get('kind') == 'table' else '논문 평균')]}


AI_STATUS = ['왕', '장군', '선비', '훈장', '의원', '어사', '역관', '상인', '화공', '악공', '농부', '장인', '포졸']


def ai_messages(reading):
    """모델에 보내는 건 이것뿐: 눈 유형 하나 + 부위별 단계 라벨 (속설 문장·사진·좌표·수치는 안 보낸다).
    작은 모델은 목록을 그대로 따라 쓰는 버릇이 있어서, 결과를 목록이 아니라 '한 문장'으로 준다.  평균이 아닌 특징에는 (특징) 을 붙인다.
    질문: 이 관상에 어울리는 과거 신분 추천, 이유는 한 줄"""
    lab = reading['labels']
    parts = []
    for k in AI_KEYS:
        it = lab['items'][k]
        parts.append('%s %s%s' % (RW_SHORT[k], it['label'], '(특징)' if it['level'] != 3 else ''))
    prompt = ('너는 재미로 보는 관상 해설가야.\n'
              '이 사람은 눈 유형이 %s이고, %s 이야.\n\n'
              '이 관상에 어울리는 과거 신분 1개를 추천해줘. (특징)을 근거로 간단한 이유를 딱 한 줄만 써.\n'
              '신분은 %s 중에서 골라. 좋다 나쁘다, 앞날을 단정하지 마.\n\n'
              '아래 형식으로 딱 한 번만 답해:\n'
              '과거 신분: (신분)\n'
              '이유: (한 줄)') % (lab['eye_shape']['ranking'][0]['name'], ', '.join(parts), ', '.join(AI_STATUS))
    return [{'role': 'user', 'content': prompt}]          # system 역할이 없는 모델도 있어서 user 하나로


def ai_clean(text):
    """모델 답을 거른다.  입력을 따라 쓴 줄(눈 유형: … / 눈꼬리: …) · 짧은 '이름: 값' 꼴의 줄 · (특징) · ★ 를 지우되,
    답 형식의 두 줄(과거 신분 / 이유)은 남긴다.  형식(신분)이 없거나 같은 줄이 되풀이되면 거절한다"""
    keep = ('과거 신분', '이유')
    lines = []
    for ln in text.splitlines():
        t = ln.strip().lstrip('-•* ').strip().replace('★', '').replace('(특징)', '').strip()
        if not t:
            continue
        if ':' in t and len(t) < 40 and not t.startswith(keep):    # '눈꼬리 낮음: 낮음', '감정: 매우 행복한 감정' 같은 줄
            continue
        lines.append(t)
    first = next((i for i, t in enumerate(lines) if t.startswith('과거 신분')), None)       # 모델이 형식을 두 번 쓰면 첫 번째만
    if first is not None:
        why = next((t for t in lines[first + 1:] if t.startswith('이유')), None)
        lines = [lines[first]] + ([why] if why else [])
    out = '\n'.join(lines).strip()
    repeated = len(lines) - len(set(lines))
    if len(out) < 10 or '신분' not in out or repeated >= 2:
        raise RuntimeError('모델이 답 형식을 못 지켰어요 (원문: %s)' % text.replace('\n', ' / ')[:80])
    return out


def ai_ask(messages, temperature=None):
    pre = _ai_preset()
    body = {'model': AI['model'], 'messages': messages, 'temperature': pre['temperature'] if temperature is None else temperature,
            'max_tokens': pre['max_tokens'], 'frequency_penalty': pre.get('frequency_penalty', 0.0), 'stream': False}
    req = urllib.request.Request(AI['url'].rstrip('/') + '/chat/completions', data=json.dumps(body).encode('utf-8'),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=AI['timeout']) as resp:
        text = json.load(resp)['choices'][0]['message']['content'] or ''
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.M).strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    return ai_clean(text)


def _ai_worker(ai, messages):
    error = None
    for temperature in (None, 0.7):                              # 형식을 못 지키면 온도를 올려 한 번 더
        try:
            ai['text'] = ai_ask(messages, temperature)
            ai['state'] = 'done'
            return
        except Exception as e:                                   # 연결 실패 · 시간 초과 · 모델 이름 오류 · 형식 불량
            error = e
            if isinstance(e, OSError):                           # 연결 문제는 다시 해도 소용없다
                break
    ai['error'] = '%s: %s' % (type(error).__name__, str(error)[:120])
    ai['state'] = 'error'


def ai_blocks(ai):
    if ai['state'] == 'waiting':
        return [{'t': 'p', 'text': '%s 가 추천을 쓰는 중이에요…' % AI['model'], 'color': 'muted'}]
    if ai['state'] == 'done':
        out = [{'t': 'p', 'text': ai['text']}]
        if ai.get('bad'):
            out.append({'t': 'p', 'text': '※ 작은 모델이 형식을 못 지켰어요. --ai-model 로 더 큰 모델(gemma2:2b, qwen2.5:1.5b)을 써 보세요.', 'size': 13, 'color': 'red'})
        return out + [{'t': 'p', 'text': '※ 눈 유형과 부위 라벨만 보고 쓴 글이에요. 틀릴 수 있어요.', 'size': 13, 'color': 'muted'}]
    return [{'t': 'p', 'text': 'AI 추천을 받지 못했어요. 기본 해설은 위에 있어요.', 'color': 'red'},
            {'t': 'p', 'text': '%s  (주소 %s · 모델 %s)' % (ai.get('error', ''), AI['url'], AI['model']), 'size': 13, 'color': 'muted'}]


def reading_to_text(r, brief=False):
    """읽을거리 → 글자만.  brief=True 는 터미널용 요약: 기본 해설 + 닮은 눈 상위 3 + 눈꼬리·두 눈 사이 현재 단계 + 부위 값.
    (설명 문장 · 단계표 · 측정값 42개 · 자세 정보는 reading.txt 에만 남는다)"""
    out = ['[%s]  눈: %s%s' % (r['title'], r['head'], r['note'].replace('  ·  ', ' · ')), '   '.join(c for c, _ in r['chips']), '']
    for title, blocks in r['sections']:
        main = title.startswith('한눈에')
        if brief and (title.startswith(('측정값 전체', '자세 · 품질', '기준과 안내', '삼정', '십이궁', '눈에 띄는')) or (title.startswith('쌍꺼풀') and not any(b['t'] == 'bars' for b in blocks))):
            continue
        out += ['■ ' + title]
        for b in blocks:
            t = b['t']
            if t == 'p':
                if not brief or main or b.get('brief'):
                    out.append('  ' + b['text'])
            elif t == 'kv':
                if not brief:
                    out += ['  %s: %s' % kv for kv in b['rows']]
            elif t == 'scale':
                if not brief:
                    out += ['  %s단계 %s (%s) - %s%s' % (lv, nm, rg, ds, '   ◀ 지금' if lv == b['current'] else '') for lv, nm, rg, ds in b['rows']]
                out.append('  ' + b.get('value_text', ''))
            elif t == 'bars':
                out += ['  %s  %s' % (n, tx) for n, _, tx in (b['rows'][:3] if brief else b['rows'])]
            elif t == 'levelrow':
                out.append('  %s · %s (%d/5)  %s' % (b['name'], b['label'], b['level'], b['caption']))
                if not brief:
                    out += ['    ' + x for x in b['extra']]
            elif t == 'table':
                if not brief:
                    out.append('  [%s]' % b['title'])
                    out += ['    %s  %s  %s' % tuple(row) for row in b['rows']]
        out.append('')
    out += r['footer']
    return '\n'.join(out)


# ---- 그리기 ---------------------------------------------------------------------------------------------------

_RW_FONTS: dict = {}


def _rw_font_path(bold):
    here = Path(__file__).resolve().parent
    env = os.environ.get('FACE_FONT_BOLD' if bold else 'FACE_FONT')
    reg = ['NanumGothic.ttf', 'NanumBarunGothic.ttf', 'NanumSquareR.ttf']
    bld = ['NanumGothicBold.ttf', 'NanumBarunGothicBold.ttf', 'NanumSquareB.ttf']
    dirs = [str(here), '/usr/share/fonts/truetype/nanum', '/usr/share/fonts/nanum', os.path.expanduser('~/.fonts'),
            os.path.expanduser('~/.local/share/fonts'), '/usr/local/share/fonts', 'C:/Windows/Fonts']
    cands = ([env] if env else []) + [os.path.join(d, n) for n in (bld if bold else reg) for d in dirs]
    cands += ['/usr/share/fonts/opentype/noto/NotoSansCJK-%s.ttc' % ('Bold' if bold else 'Regular'),
              '/usr/share/fonts/truetype/noto/NotoSansCJK-%s.ttc' % ('Bold' if bold else 'Regular'),
              'C:/Windows/Fonts/%s' % ('malgunbd.ttf' if bold else 'malgun.ttf'), '/System/Library/Fonts/AppleSDGothicNeo.ttc']
    for c in cands:
        if c and os.path.exists(c):
            return c
    import glob
    found = sorted(glob.glob('/usr/share/fonts/**/Nanum*Gothic*.ttf', recursive=True))
    found = [f for f in found if ('Bold' in f) == bool(bold) and 'Coding' not in f] or found
    if found:
        return found[0]
    try:
        out = subprocess.run(['fc-list', ':lang=ko', 'file'], capture_output=True, text=True, timeout=5).stdout
        files = [ln.split(':')[0].strip() for ln in out.splitlines() if ln.strip()]
        if files:
            return files[0]
    except Exception:
        pass
    if bold:
        return _rw_font_path(False)
    raise FileNotFoundError('한글 글꼴을 찾지 못했어요. NanumGothic.ttf 를 이 파일 옆에 두거나 FACE_FONT=/경로/NanumGothic.ttf 로 지정하세요.')


def _rw_font(size, bold=False):
    key = (size, bool(bold))
    if key not in _RW_FONTS:
        _RW_FONTS[key] = ImageFont.truetype(_rw_font_path(bold), size)
    return _RW_FONTS[key]


def _rw_wrap(text, font, max_w):
    lines = []
    for para in str(text).split('\n'):
        if not para.strip():
            lines.append('')
            continue
        cur = ''
        for word in para.split(' '):
            trial = word if not cur else cur + ' ' + word
            if font.getlength(trial) <= max_w:
                cur = trial
                continue
            if cur:
                lines.append(cur)
                cur = ''
            while font.getlength(word) > max_w:                 # 너무 긴 단어는 글자 단위로 자른다
                k = len(word)
                while k > 1 and font.getlength(word[:k]) > max_w:
                    k -= 1
                lines.append(word[:k])
                word = word[k:]
            cur = word
        lines.append(cur)
    return lines


def _rw_rrect(d, box, radius, fill=None, outline=None, width=1):
    try:
        d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
    except AttributeError:                                       # 오래된 Pillow
        d.rectangle(box, fill=fill, outline=outline, width=width)


def _rw_chip(d, x, y, text, font, outline, fg):
    w = int(font.getlength(text)) + 24
    h = font.size + 14
    _rw_rrect(d, (x, y, x + w, y + h), h // 2, fill=(24, 20, 34), outline=outline, width=2)
    d.text((x + 12, y + 6), text, font=font, fill=fg)
    return w, h


def _rw_blocks(d, blocks, W):
    """blocks 를 위에서부터 그리고 마지막 y 를 돌려준다 (d 가 작은 더미 그림이면 높이만 재는 셈)"""
    y = 4
    F = _rw_font
    for b in blocks:
        t = b['t']
        if t == 'h2':
            y += 14
            d.rectangle((0, y + 2, 5, y + 26), fill=RW_ACCENT)
            d.text((16, y), b['text'], font=F(22, True), fill=RW_ACCENT)
            y += 36
            d.line((0, y, W, y), fill=RW_BORDER, width=1)
            y += 12
        elif t == 'p':
            size = b.get('size', 18)
            font = F(size, b.get('bold', False))
            color = RW_COLORS.get(b.get('color', 'text'), RW_TEXT)
            for line in _rw_wrap(b['text'], font, W - 4):
                d.text((0, y), line, font=font, fill=color)
                y += size + 9
            y += b.get('after', 6)
        elif t == 'kv':
            cols = b.get('cols', 2)
            cw = W / cols
            rows = b['rows']
            for i in range(0, len(rows), cols):
                for j in range(cols):
                    if i + j < len(rows):
                        k, v = rows[i + j]
                        x0 = j * cw + (14 if j else 0)
                        x1 = (j + 1) * cw - (14 if j < cols - 1 else 0)
                        d.text((x0, y), k, font=F(16), fill=RW_MUTED)
                        d.text((x1, y), str(v), font=F(16, True), fill=RW_TEXT, anchor='ra')
                y += 28
            y += 6
        elif t == 'scale':
            xs = [8, 62, 214, 392]
            desc_w = W - xs[3] - 10
            for x_, name in zip(xs, ('단계', '이름', '범위', '설명')):
                d.text((x_, y), name, font=F(14), fill=RW_MUTED)
            y += 24
            for lv, name, rng, desc in b['rows']:
                lines = _rw_wrap(desc, F(16), desc_w)
                rh = max(36, len(lines) * 23 + 12)
                if lv == b['current']:
                    _rw_rrect(d, (0, y, W, y + rh - 4), 8, fill=(58, 46, 24), outline=RW_ACCENT, width=2)
                d.rectangle((8, y + 8, 14, y + rh - 12), fill=RW_LEVEL[lv - 1])
                d.text((xs[0] + 14, y + 7), str(lv), font=F(17, True), fill=RW_TEXT)
                d.text((xs[1], y + 7), name, font=F(17, True), fill=RW_LEVEL[lv - 1] if lv == b['current'] else RW_TEXT)
                d.text((xs[2], y + 8), rng, font=F(16), fill=RW_TEXT)
                for k, ln in enumerate(lines):
                    d.text((xs[3], y + 8 + k * 23), ln, font=F(16), fill=RW_MUTED if lv != b['current'] else RW_TEXT)
                y += rh
            y += 4
            if b.get('value_text'):
                d.text((0, y), b['value_text'], font=F(19, True), fill=RW_ACCENT)
                y += 36
        elif t == 'bars':
            name_w, bar_x = 130, 142
            bar_w = W - bar_x - 70
            for name, frac, text in b['rows']:
                d.text((0, y), name, font=F(17), fill=RW_TEXT)
                _rw_rrect(d, (bar_x, y + 8, bar_x + bar_w, y + 20), 6, fill=RW_CARD2)
                fw = int(bar_w * max(0.0, min(1.0, frac)))
                if fw > 4:
                    _rw_rrect(d, (bar_x, y + 8, bar_x + fw, y + 20), 6, fill=RW_ACCENT)
                d.text((W, y), text, font=F(16, True), fill=RW_TEXT, anchor='ra')
                y += 30
            y += 6
        elif t == 'levelrow':
            lc = RW_LEVEL[b['level'] - 1]
            d.text((0, y), b['name'], font=F(17), fill=RW_MUTED)
            d.text((W, y - 2), b['label'], font=F(21, True), fill=lc, anchor='ra')
            y += 32
            seg_w = (W - 4 * 6) / 5
            for s in range(5):
                x0 = int(s * (seg_w + 6))
                d.rectangle((x0, y, int(x0 + seg_w), y + 10), fill=lc if s == b['level'] - 1 else RW_CARD2)
            y += 18
            d.text((0, y), b['caption'], font=F(14), fill=RW_MUTED)
            y += 22
            for ex in b['extra']:
                for ln in _rw_wrap(ex, F(16), W - 4):
                    d.text((0, y), ln, font=F(16), fill=RW_TEXT)
                    y += 24
            if b.get('caveat'):
                d.text((0, y), '※ ' + b['caveat'], font=F(13), fill=RW_MUTED)
                y += 20
            y += 14
        elif t == 'table':
            if b.get('title'):
                d.text((0, y), b['title'], font=F(17, True), fill=RW_ACCENT)
                y += 28
            xs, acc = [], 0.0
            for wf in b['widths']:
                xs.append(acc * W)
                acc += wf
            xs.append(W)
            def cell(i, text, yy, font, color):
                a = b['aligns'][i]
                if a == 'r':
                    d.text((xs[i + 1] - 12, yy), text, font=font, fill=color, anchor='ra')
                else:
                    d.text((xs[i] + 10, yy), text, font=font, fill=color)
            for i, name in enumerate(b['head']):
                cell(i, name, y, F(14), RW_MUTED)
            y += 22
            for r_i, row in enumerate(b['rows']):
                if r_i % 2 == 0:
                    d.rectangle((0, y - 2, W, y + 24), fill=RW_CARD2)
                for i, text in enumerate(row):
                    cell(i, text, y, F(16, i == 1), RW_TEXT if i < 2 else RW_MUTED)
                y += 27
            y += 12
        elif t == 'gap':
            y += b.get('h', 10)
    return y + 8


def render_reading_doc(reading, W, min_h):
    """읽을거리 본문 → 긴 그림 (너비 W, 높이는 내용만큼. min_h 보다 짧으면 min_h 로 채운다 - 모자란 곳이 까맣게 되지 않게)"""
    blocks = []
    for title, bl in reading['sections']:
        blocks.append({'t': 'h2', 'text': title})
        blocks += bl
    probe = ImageDraw.Draw(Image.new('RGB', (4, 4)))
    height = _rw_blocks(probe, blocks, W)
    img = Image.new('RGB', (W, max(height, min_h)), RW_CARD)
    _rw_blocks(ImageDraw.Draw(img), blocks, W)
    return img, height


def render_reading_frame(reading, doc_img, scroll):
    """고정 부분(머리 · 요약 카드 · 아래 안내) + 본문 scroll 만큼. 반환: (BGR 그림, 최대 스크롤, 적용된 스크롤)"""
    F = _rw_font
    base, body_top, body_bottom = _rw_static(reading)
    view_h = body_bottom - body_top - 24
    max_scroll = max(0, doc_img.height - view_h)
    scroll = max(0, min(int(scroll), max_scroll))
    frame = base.copy()
    frame.paste(doc_img.crop((0, scroll, doc_img.width, scroll + view_h)), (RW_PAD + 22, body_top + 12))
    if max_scroll > 0:                                           # 스크롤바
        d = ImageDraw.Draw(frame)
        bar_h = max(36, int(view_h * view_h / doc_img.height))
        bar_y = body_top + 12 + int((view_h - bar_h) * scroll / max_scroll)
        d.rectangle((RW_W - RW_PAD - 11, bar_y, RW_W - RW_PAD - 5, bar_y + bar_h), fill=RW_BORDER)
        d.text((RW_W - RW_PAD - 8, body_bottom + 8), '%d%%' % int(100 * scroll / max_scroll), font=F(13), fill=RW_MUTED, anchor='ra')
    return cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR), max_scroll, scroll


_RW_STATIC: dict = {}


def _rw_static(reading):
    key = id(reading)
    if _RW_STATIC.get('key') == key:
        return _RW_STATIC['v']
    F = _rw_font
    img = Image.new('RGB', (RW_W, RW_H), RW_BG)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, RW_W, 64), fill=(34, 18, 48))
    d.rectangle((0, 60, RW_W, 64), fill=RW_ACCENT)
    d.text((RW_PAD, 14), reading['title'], font=F(30, True), fill=RW_ACCENT)
    d.text((RW_W - RW_PAD, 24), 'FACE FORTUNE', font=F(15), fill=RW_MUTED, anchor='ra')

    # 요약 카드: 눈 유형 + 칩
    y0 = 80
    inner_w = RW_W - RW_PAD * 2 - 44
    chips_h = F(16).size + 14 + 8
    def chip_rows(draw_it):
        x, y = 0, 0
        for text, hot in reading['chips']:
            w = int(F(16).getlength(text)) + 24
            if x + w > inner_w:
                x, y = 0, y + chips_h
            if draw_it:
                _rw_chip(d, RW_PAD + 22 + x, cy + y, text, F(16), RW_ACCENT if hot else RW_BORDER, RW_TEXT if hot else RW_MUTED)
            x += w + 8
        return y + chips_h
    cy = 0
    n_h = chip_rows(False)
    card_h = 20 + 44 + n_h + 10
    d.rectangle((RW_PAD, y0, RW_W - RW_PAD, y0 + card_h), fill=RW_CARD, outline=RW_BORDER, width=1)
    d.rectangle((RW_PAD, y0, RW_PAD + 6, y0 + card_h), fill=RW_ACCENT)
    x0 = RW_PAD + 22
    d.text((x0, y0 + 20), '눈', font=F(18, True), fill=RW_MUTED)
    d.text((x0 + 34, y0 + 12), reading['head'], font=F(30, True), fill=RW_ACCENT)
    d.text((x0 + 34 + int(F(30, True).getlength(reading['head'])), y0 + 22), reading['note'], font=F(16), fill=RW_TEXT)
    cy = y0 + 66
    chip_rows(True)

    body_top = y0 + card_h + 14
    foot_h = 80
    body_bottom = RW_H - foot_h
    d.rectangle((RW_PAD, body_top, RW_W - RW_PAD, body_bottom), fill=RW_CARD, outline=RW_BORDER, width=1)
    fy = body_bottom + 8
    for k, line in enumerate(reading['footer']):
        d.text((RW_PAD, fy + k * 21), line, font=F(15), fill=RW_GREEN if k == 0 else RW_MUTED)
    d.text((RW_PAD, fy + 46), '↑↓ · 마우스 휠 · PageUp/PageDown 스크롤     Home/End 처음·끝     Q 닫기', font=F(14), fill=RW_MUTED)
    _RW_STATIC.update(key=key, v=(img, body_top, body_bottom))
    return _RW_STATIC['v']


_RW_UP = {ord('w'), ord('k'), 65362, 2490368}
_RW_DOWN = {ord('s'), ord('j'), 65364, 2621440}
_RW_PGUP, _RW_PGDN = {65365, 2162688}, {65366, 2228224}
_RW_HOME, _RW_END = {65360, 2359296}, {65367, 2293760}


def simple_reading(r):
    """창에는 이것만: 눈 유형·칩(맨 위 카드) + 한눈에 보기 + 종합 등급 + 얼굴형(오행).  나머지는 reading.txt 에만 남는다"""
    keep = ('한눈에', '종합 등급', '얼굴형 (오행)', '운세 7가지')
    secs = []
    for title, blocks in r['sections']:
        if title.startswith(keep):
            secs.append((title, blocks if title.startswith(('한눈에', '운세 7가지')) else [b for b in blocks if b.get('brief')]))
    return dict(r, sections=secs, footer=['※ 재미로 보는 관상이에요. 과학적으로 검증된 분석이 아니에요.'])


def run_result_window(folder, reference=None, sex='any', title='Face Reading', ai_on=False, use_window=True):
    """저장된 캡처 폴더의 관상 풀이를 새 창으로 보여준다. 창이 없으면(DISPLAY 없음) 글자로 출력한다"""
    folder = Path(folder)
    try:
        analysis = json.loads((folder / 'analysis_result.json').read_text(encoding='utf-8'))
    except Exception as error:
        print('analysis_result.json 을 읽지 못했어요:', folder, error)
        return 1
    vision = None
    try:
        vision = json.loads((folder / 'vision_landmarks.json').read_text(encoding='utf-8'))
        vision.pop('landmarks', None)
    except Exception:
        pass
    reading = build_reading(analysis, vision, folder.name, reference, sex)
    if reading is None:
        print('이 캡처에는 측정값(feature_values_ipd)이 없어서 풀이를 만들 수 없어요:', folder)
        return 1
    ai = {'state': 'waiting', 'text': '', 'error': ''}
    if ai_on:
        threading.Thread(target=_ai_worker, args=(ai, ai_messages(reading)), daemon=True).start()
    text = reading_to_text(reading)
    try:
        (folder / 'reading.txt').write_text(text + '\n', encoding='utf-8')
    except OSError:
        pass
    has_display = bool(os.environ.get('DISPLAY')) or not sys.platform.startswith('linux')
    if not use_window or not has_display or Image is None:       # 터미널 출력 (--terminal 이거나 창을 못 열 때)
        print('\n' + '=' * 64)
        print(reading_to_text(reading, brief=True))
        if ai_on:
            print('\n■ AI 추천 (과거 신분) (%s) 쓰는 중…' % AI['model'])
            while ai['state'] == 'waiting':
                time.sleep(0.2)
            print('\n'.join(b['text'] for b in ai_blocks(ai)))
            if ai['state'] == 'done':
                try:
                    (folder / 'reading.txt').write_text(text + '\n\n■ AI 추천 (과거 신분)\n' + ai['text'] + '\n', encoding='utf-8')
                except OSError:
                    pass
        if use_window and Image is None:
            print('\n(pillow 가 없어 글자로만 보여드려요: pip install pillow)')
        print('\n(전체 내용은 %s)\n' % (folder / 'reading.txt') + '=' * 64)
        return 0
    try:
        win = simple_reading(reading)
        if ai_on:
            win['sections'].insert(0, ('AI 추천 (과거 신분)', ai_blocks(ai)))
        _rw_static(win)                                          # 글꼴이 없으면 여기서 FileNotFoundError
        view_h = _RW_STATIC['v'][2] - _RW_STATIC['v'][1] - 24
        doc_img, _ = render_reading_doc(win, RW_W - RW_PAD * 2 - 44 - 14, view_h)
    except FileNotFoundError as error:
        print(error)
        print(text)
        return 1

    state = {'scroll': 0, 'dirty': True}

    def on_mouse(event, x, y, flags, param):
        if event == getattr(cv2, 'EVENT_MOUSEWHEEL', 10):
            try:
                delta = cv2.getMouseWheelDelta(flags)
            except Exception:
                delta = flags
            state['scroll'] -= 60 if delta > 0 else -60
            state['dirty'] = True

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, on_mouse)
    shown_ai = 'waiting'
    while True:
        if ai_on and ai['state'] != shown_ai:                    # 답이 오면 같은 창의 맨 위 구역만 바꿔 다시 그린다
            shown_ai = ai['state']
            win['sections'][0] = ('AI 추천 (과거 신분)', ai_blocks(ai))
            doc_img, _ = render_reading_doc(win, RW_W - RW_PAD * 2 - 44 - 14, view_h)
            state['dirty'] = True
            if shown_ai == 'done':
                try:
                    (folder / 'reading.txt').write_text(text + '\n\n■ AI 추천 (과거 신분)\n' + ai['text'] + '\n', encoding='utf-8')
                except OSError:
                    pass
        if state['dirty']:
            frame, max_scroll, state['scroll'] = render_reading_frame(win, doc_img, state['scroll'])
            state['dirty'] = False
        cv2.imshow(title, frame)
        k = cv2.waitKeyEx(50)
        if k in (ord('q'), ord('Q'), 27):
            break
        step = 0
        if k in _RW_UP:
            step = -50
        elif k in _RW_DOWN:
            step = 50
        elif k in _RW_PGUP:
            step = -380
        elif k in _RW_PGDN:
            step = 380
        elif k in _RW_HOME:
            step = -10 ** 6
        elif k in _RW_END:
            step = 10 ** 6
        if step:
            state['scroll'] += step
            state['dirty'] = True
        try:
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
    cv2.destroyAllWindows()
    return 0


# ---- 촬영 프로그램에서 창 띄우기 (따로 도는 프로세스) ----------------------------------------------------------

LABEL_UI: dict[str, Any] = {"labels": None, "folder": None, "proc": None, "open_request": False}


def open_result_window(folder, extra_args=(), window=True, reference=None, sex='any', ai_on=False):
    """저장된 캡처의 관상 풀이를 보여준다.  기본은 새 창(따로 도는 프로세스라 촬영이 멈추지 않는다).  window=False 면 터미널 출력(스레드)"""
    if not window:
        threading.Thread(target=run_result_window, args=(folder, reference, sex), kwargs={'ai_on': ai_on, 'use_window': False}, daemon=True).start()
        return
    if os.environ.get('DISPLAY') is None and sys.platform.startswith('linux'):      # 창을 못 열면 터미널로
        threading.Thread(target=run_result_window, args=(folder, reference, sex), kwargs={'ai_on': ai_on, 'use_window': False}, daemon=True).start()
        return
    old = LABEL_UI.get('proc')
    if old is not None and old.poll() is None:
        old.terminate()
    try:
        LABEL_UI['proc'] = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--view', str(folder), *extra_args])
    except Exception as error:
        print('풀이 창을 열지 못했어요:', error)


def _ktext_w(text, size):
    if KOREAN_FONT is None:
        return int(len(text) * size * 0.62)
    return int(_kfont(size).getlength(text))


def draw_gwansang_panel(panel, y, h):
    """오른쪽 패널의 [관상] 영역: 스캔 중에는 실시간 라벨, 끝나면 최종 라벨.  자세한 내용은 새 창에서"""
    x1, x2 = SEG_X, PANEL_W - SEG_X
    px, rx = x1 + 22, x2 - 22
    lab = LABEL_UI.get("labels")
    cv2.circle(panel, (px + 5, y + 30), 5, C_MINT, -1, cv2.LINE_AA)
    draw_ktext(panel, "관상", px + 22, y + 16, 22, C_TEXT, "GWANSANG")
    if lab is None:
        draw_ktext(panel, "측정이 시작되면 표시돼요", px, y + 76, 16, C_MUTED, "Shown while scanning")
        draw_ktext(panel, "스캔이 끝나면 풀이 창이 새로 열려요", px, y + 106, 14, C_DIM, "A result window opens when the scan ends")
        return
    es = lab["eye_shape"]
    top = es["ranking"][0]
    draw_ktext(panel, "눈  %s  %d%%" % (top["name"], int(top["similarity"] * 100)), px, y + 52, 22, C_MINT if es["distinct"] else C_MUTED,
               "EYE  %s  %d%%" % (top["name_en"].upper(), int(top["similarity"] * 100)))
    ry = y + 90
    if len(LABEL_ORDER) > 11:                                     # 라벨이 많으면 두 칸으로 이름 · 단계만
        col_w = (rx - px) // 2
        for i_, key in enumerate(LABEL_ORDER):
            it = lab["items"][key]
            cx, cy = px + (i_ % 2) * (col_w + 6), ry + (i_ // 2) * 44
            draw_ktext(panel, RW_SHORT[key], cx, cy, 13, C_MUTED, it["name_en"])
            draw_ktext(panel, it["label"] + (" ?" if it.get("reliability") == "low" else ""), cx, cy + 17, 16,
                       LEVEL_COLORS[it["level"] - 1] if it["level"] != 3 else C_TEXT, it["label_en"])
        draw_ktext(panel, "자세한 풀이 · 운세는 풀이 창에서  (V 키 / [관상] 버튼)", px, y + h - 30, 13, C_MUTED, "V key -> result window")
        return
    compact = len(LABEL_ORDER) > 8
    for key in LABEL_ORDER:
        it = lab["items"][key]
        lv = it["level"] - 1
        color = LEVEL_COLORS[lv]
        draw_ktext(panel, it["name_ko"], px, ry, 15, C_MUTED, it["name_en"])
        w = _ktext_w(it["label"], 18)
        draw_ktext(panel, it["label"], rx - w, ry - 2, 18, color, it["label_en"])
        seg = (rx - px - 4 * 4) / 5
        for s_ in range(5):
            sx = int(px + s_ * (seg + 4))
            cv2.rectangle(panel, (sx, ry + (22 if compact else 24)), (int(sx + seg), ry + (28 if compact else 30)), color if s_ == lv else C_SURFACE2, -1)
        if not compact:
            draw_ktext(panel, it["caption"] + ("  ?" if it.get("reliability") == "low" else ""), px, ry + 33, 12, C_DIM, it["caption"])
        elif it.get("reliability") == "low":
            draw_ktext(panel, "?", rx + 6, ry, 15, C_AMBER, "?")
        ry += 36 if compact else 50
    draw_ktext(panel, "자세한 풀이 · 속설 · 측정값은  V 키 / [관상] 버튼  →  새 창", px, y + h - 30, 13, C_MUTED, "V key / GWANSANG button -> result window")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--headless", action="store_true", help="창 없이 (SSH 테스트용)")
    parser.add_argument("--max-frames", type=int, default=0, help="이 프레임 수 뒤 종료 (0=무제한)")
    parser.add_argument("--output-dir", type=Path, default=CAPTURE_DIR)
    parser.add_argument("--no-hairline", action="store_true", help="헤어라인 분할 끄기 (가상 이마 점만 사용)")
    parser.add_argument("--no-eye-refine", action="store_true", help="2단계 눈 정밀 모델 끄기 (1단계 눈 점 그대로)")
    parser.add_argument("--no-eyelid", action="store_true", help="쌍꺼풀/홑꺼풀 판정 끄기")
    parser.add_argument("--sex", choices=["f", "m", "any"], default="any", help="(논문 기준일 때만) 코 너비 기준을 성별에 맞춤")
    parser.add_argument("--reference", type=Path, help="5단계 기준표 CSV (기본: 이 파일 옆의 face_reference*.csv)")
    parser.add_argument("--no-reference", action="store_true", help="기준표를 쓰지 않고 문서의 논문 평균±편차로 라벨")
    parser.add_argument("--no-ai", action="store_true", default=True, help=argparse.SUPPRESS)      # 이 버전은 AI 를 쓰지 않는다
    parser.add_argument("--terminal", action="store_true", help="관상 풀이를 새 창 대신 터미널에 출력")
    parser.add_argument("--no-window", action="store_true", help="스캔이 끝나도 관상 풀이를 자동으로 보여주지 않음 (V 키로는 보임)")
    parser.add_argument("--view", type=Path, metavar="CAPTURE_DIR", help="촬영 없이 저장된 캡처의 관상 풀이 창만 연다")
    args = parser.parse_args()
    args.no_ai, args.ai_model, args.ai_url = True, None, None          # Qwen 을 쓰지 않는 버전

    reference = None if args.no_reference else find_reference(args.reference)
    if args.ai_model: AI["model"] = args.ai_model
    if args.ai_url: AI["url"] = args.ai_url
    extra = (["--no-ai"] if args.no_ai else []) + (["--ai-model", args.ai_model] if args.ai_model else []) + (["--ai-url", args.ai_url] if args.ai_url else []) \
            + ["--sex", args.sex] + (["--no-reference"] if args.no_reference else (["--reference", str(args.reference)] if args.reference else []))
    if args.view is not None:                                    # 풀이 창만 (카메라·모델 없이)
        sys.exit(run_result_window(args.view, reference, args.sex, ai_on=not args.no_ai, use_window=not args.terminal))

    landmarker = Landmarker(eye_refine=not args.no_eye_refine)
    hair = Hairline(enabled=not args.no_hairline)
    eyelid = EyelidClassifier(enabled=not args.no_eyelid)
    print("Labels reference:", ("table %s (n=%d)" % (Path(reference.path).name, min(s["count"] for s in reference.stats.values()))
                                if reference else "paper mean+-SD (%s)" % ("--no-reference" if args.no_reference else "no face_reference*.csv found")))
    print("Eyelid:", "on (%s, %dx%d)" % (eyelid.meta.get("model", "model"), *eyelid.size) if eyelid.ok else "off - " + eyelid.reason)
    cap = open_camera(args.camera)

    measurement = Measurement()
    coordinates: list[np.ndarray] = []
    analyses: list[dict[str, Any]] = []
    eyelid_samples: list[dict[str, float]] = []
    label_state: dict[str, Any] = {"m": None, "s": None, "n": 0}     # 지금까지 모은 프레임의 평균·표준편차·개수 (실시간 라벨용)
    save_state: dict[str, Any] = {"path": None, "saved_at": None, "error": None}
    ready_screen = ready_frame = None
    last_hairline = None
    invalid_streak = frames = 0

    def reset_scan() -> None:
        nonlocal ready_screen, ready_frame, invalid_streak, last_hairline
        measurement.reset(); coordinates.clear(); analyses.clear()
        eyelid_samples.clear(); label_state.update(m=None, s=None, n=0); LABEL_UI["labels"] = None
        save_state.update(path=None, saved_at=None, error=None)
        ready_screen = ready_frame = None; last_hairline = None; invalid_streak = 0

    def show_reading(folder):
        open_result_window(folder, extra, not args.terminal and not args.headless, reference, args.sex, not args.no_ai)

    def do_save(frame_for_save, screen_for_save, tag):
        try:
            final_eyelid = eyelid.aggregate(eyelid_samples, final=True)
            final_ipd = {k[5:]: v for k, v in measurement.mean.items() if k.startswith("ipd__")}
            final_std = {k[5:]: v for k, v in measurement.std.items() if k.startswith("ipd__")}
            final_labels = face_labels(final_ipd, final_eyelid, args.sex, reference, final_std, len(measurement.samples), eye_tilts(aggregate_eyes(analyses)))   # 30프레임 평균으로 최종 라벨
            LABEL_UI["labels"] = final_labels
            p = save_capture(args.output_dir, coordinates, analyses, frame_for_save, screen_for_save, measurement,
                             eyelid=final_eyelid, labels=final_labels, landmark_info=landmarker.info())
            save_state.update(path=p, saved_at=time.monotonic(), error=None)
            LABEL_UI["folder"] = p
            v = measurement.mean
            print(f"{tag}: {p}")
            print("  face_hw=%.3f cheek/jaw=%.3f jaw_angle=%.1f nose_hw=%.3f mouth_w=%.3f  (cv=%.4f)"
                  % (v["face_hw_ratio"], v["cheek_to_jaw_width_ratio"], v["jaw_angle_deg"], v["nose_hw_ratio"], v["mouth_width_ratio"], measurement.score))
            if final_labels:
                es = final_labels["eye_shape"]
                print("  eye: %s %d%%  |  " % (es["best"] if es["distinct"] else "no clear type", int(es["ranking"][0]["similarity"] * 100))
                      + "  ".join("%s=%s" % (it["name_en"], it["label_en"]) for it in final_labels["items"].values()))
            if not args.no_window:
                show_reading(p)                                                           # 관상 풀이 창 (--terminal 이면 터미널)
        except OSError as error:
            save_state.update(error=str(error)); print("Save failed:", error)

    print(f"Collecting {TARGET_FRAMES} frames per scan. Keys: R new / S save / V result window / 1 gwansang / 2 lookalike / Q quit")
    if KOREAN_FONT is None:
        print("Korean font not found; using English button labels (put NanumGothic.ttf next to this file to fix)")
    if not args.headless:
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, on_mouse)                  # 버튼 클릭
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
                else:               status = "COLLECTING"

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
                    if eyelid.ok and len(analyses) % EYELID_EVERY == 0 and len(eyelid_samples) < EYELID_MAX_SAMPLES:
                        sample = eyelid.predict(frame, pts)                 # 거울 프레임 + 점 → 쌍꺼풀 확률
                        if sample is not None:
                            eyelid_samples.append(sample)
                    if len(measurement.samples) >= 3:                       # 지금까지의 평균으로 라벨 (스캔 중 실시간)
                        keys = [k for k in measurement.samples[0] if k.startswith("ipd__")]
                        label_state["m"] = {k[5:]: float(np.mean([s[k] for s in measurement.samples])) for k in keys}
                        label_state["s"] = {k[5:]: float(np.std([s[k] for s in measurement.samples])) for k in keys}
                        label_state["n"] = len(measurement.samples)
                if after < before:                                         # 중심이 튀어 reset 된 경우
                    del coordinates[:]; del analyses[:]
                    eyelid_samples.clear(); label_state.update(m=None, s=None, n=0); LABEL_UI["labels"] = None
                    if after == 1:
                        coordinates.append(pts.copy()); analyses.append(analysis)
                    ready_screen = ready_frame = None
                    save_state.update(path=None, saved_at=None, error=None)
                if after == TARGET_FRAMES:
                    status = "READY - AUTO SAVING" if measurement.stable else "UNSTABLE - R TO RETRY"

            if label_state["m"] is not None and save_state["path"] is None:      # 저장 전에는 실시간 라벨
                live_eyelid = eyelid.aggregate(eyelid_samples) if eyelid.ok else None
                live_tilts = {sd_: float(np.mean([a_["eyes"][sd_]["outer_corner_tilt_deg"] for a_ in analyses])) for sd_ in EYES} if analyses else None
                LABEL_UI["labels"] = face_labels(label_state["m"], live_eyelid, args.sex, reference, label_state["s"], label_state["n"], live_tilts)

            saved_age = None
            if save_state["saved_at"] is not None:
                saved_age = time.monotonic() - save_state["saved_at"]
                if save_state["path"] is not None:
                    status = "AUTO-SAVED - PRESS R FOR NEW SCAN"

            display = draw(frame, pts, analysis, status, len(measurement.samples), last_hairline if valid else None,
                           save_state["path"], saved_age, save_state["error"])

            if completed or (len(measurement.samples) == TARGET_FRAMES and measurement.stable):
                ready_screen, ready_frame = display.copy(), frame.copy()

            if (len(measurement.samples) == TARGET_FRAMES and measurement.stable and ready_screen is not None
                    and save_state["path"] is None and save_state["error"] is None):
                do_save(ready_frame, ready_screen, "Auto-saved")
                display = draw(frame, pts, analysis, "AUTO-SAVED - PRESS R FOR NEW SCAN" if save_state["path"] else "SAVE FAILED",
                               len(measurement.samples), None, save_state["path"], 0.0, save_state["error"])

            if LABEL_UI.pop("open_request", False):                            # [관상] 버튼 / V 키 → 풀이 창
                if LABEL_UI.get("folder") is not None:
                    show_reading(LABEL_UI["folder"])
                else:
                    print("아직 저장된 측정이 없어요. 스캔이 끝나면 풀이 창이 열려요.")

            if not args.headless:
                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"): break
                if key == ord("r"): reset_scan()
                if key == ord("v"): LABEL_UI["open_request"] = True
                for b in MODE_BUTTONS:                               # 키보드로도 버튼 선택 (1 관상 / 2 닮은꼴)
                    if key == ord(b["key"]): select_mode(b["id"])
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
