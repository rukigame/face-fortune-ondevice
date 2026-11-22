# 기여 범위 (Contributions)

4인 팀 프로젝트(2026.09.30 ~ 10.07). 팀원 동의를 받아 전체 코드를 공개하며,
포트폴리오에서 다루는 것은 **김진환 담당 파트(카메라 → 수치 비전 파이프라인)** 입니다.

| 경로 | 내용 | 작성 |
| --- | --- | --- |
| src/face_fortune.py — YoloFaceDetector, WflwMesh, add_virtual_points, EyeRefiner, Segmenter/Hairline, compute_metrics, Measurement(30프레임·절사 평균), 품질 검사(pose/gaze/eyes) | 검출 · 랜드마크 · 눈 보정 · 헤어라인 · 42개 비율 · 품질 관리 | **김진환** |
| training/FaceDet_Train.ipynb | YOLO11n 얼굴 검출 학습 · INT8 변환 | **김진환** |
| training/WFLW_Train.ipynb, wflw_prep.py, wflw_mesh.py | 98점 랜드마크 학습 | **김진환** |
| training/HairSeg_Train.ipynb, hairline.py | 헤어라인 분할 학습 | **김진환** |
| training/EyeRefine_Train.ipynb, eye_prep.py | 눈 보정 2단계 모델 학습 | **김진환** |
| training/Norms_Train.ipynb, face_metrics.py | 42개 비율 정의 · 기준 분포 | **김진환** |
| src/face_fortune.py — 기준표·저장·화면 그리기 일부, src/web_app.py, templates/, static/ | 웹앱 · UI | 팀원 |
| src/fortune_bridge.py, fortune_story_composer.py, face_fortune_labels18.py, gwansang_rules.json, face_reference_10level_clean.csv | 관상 해석 · 기준표 | 팀원 |
| src/run_team_animal_test.py | 동물상 · 포켓몬 매칭 | 팀원 |

※ 팀원별 세부 담당은 확인 후 이름을 채우세요. 모델 파일(.tflite)과 동물·포켓몬 이미지는 저작권·용량 문제로 저장소에 포함하지 않습니다.
