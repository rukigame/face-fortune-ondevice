# =====================================================================
#  face_metrics.py  -  WFLW 98점(+가상 점 5개) → 관상용 측정값   [FaceReading_3 / 방법 A]
#  Colab(기준표 생성)과 보드(실시간)에서 같은 파일을 그대로 씀  !! colab/ 과 board/ 가 동일해야 함
#
#  - 입력 pts: wflw_mesh.WflwMesh 가 돌려주는 (103,3) 픽셀 좌표 (z 는 0)
#  - 모든 길이는 두 눈동자(96,97) 사이 거리(IPD)로 나눈 비율, 각도는 도(°)
#  - MediaPipe 판과 다른 점: z 가 없어 cheek_prom / nose_tip_h / forehead_tilt 삭제,
#    콧대 양옆 점이 없어 bridge_w 삭제 → 측정값 46 → 42개
# =====================================================================
import math
import numpy as np

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
        from wflw_prep import add_virtual_points
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
