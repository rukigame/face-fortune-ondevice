# =====================================================================
#  hairline.py  -  우리가 학습한 분할 모델(hair_seg.tflite, CelebAMask-HQ 4클래스)로 헤어라인(진짜 이마 상단) 찾기
#  Colab(Norms_Train)과 보드 공용.  mediapipe 불필요 — wflw_mesh._interpreter (LiteRT) 로 실행
#
#  hair_seg.tflite 는 픽셀마다 0 배경 / 1 얼굴 피부 / 2 머리카락 / 3 모자 를 붙여 준다 (HairSeg_Train.ipynb CLASSES).
#  얼굴 피부(1)가 위로 끝나는 곳이 헤어라인이다. 파일이 없으면 Segmenter.ok == False → 가상 이마 점 사용.
# =====================================================================
import os
import numpy as np
import cv2

MODEL_FILE = 'hair_seg.tflite'
FACE_SKIN, HAIR, HAT = 1, 2, 3
SEG_SIZE = 256


class Segmenter:
    def __init__(self, path=MODEL_FILE):
        self.ok = False
        from wflw_mesh import model_path
        path = model_path(path)
        if not os.path.exists(path):
            print('%s 없음 → 헤어라인 미사용 (가상 이마 점 사용)' % path); return
        try:
            from wflw_mesh import _interpreter
            self.it = _interpreter(path)
            self.inp = self.it.get_input_details()[0]; self.out = self.it.get_output_details()[0]
            self.ok = True; print('헤어라인 분할 모델 로딩 완료:', path)
        except Exception as e:
            print('분할 모델 로딩 실패 → 헤어라인 미사용:', e)

    def mask(self, rgb):
        """rgb (H,W,3) → (H,W) uint8 클래스 맵, 실패 시 None"""
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
