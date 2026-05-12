# Korean TTS Fine-tuning (GPT-SoVITS v2 → ONNX → Mobile)

한국어 LLM-style TTS(GPT-SoVITS v2)를 파인튜닝하고, ONNX로 변환해 모바일 앱에
**온디바이스 모델**로 탑재하기 위한 작업 저장소입니다.

| 항목 | 값 |
|---|---|
| 베이스 모델 | [GPT-SoVITS v2](https://github.com/RVC-Boss/GPT-SoVITS) (GPT(AR) + SoVITS Decoder) |
| 타겟 언어 | 한국어 |
| 학습 환경 | Linux + NVIDIA RTX 4070 Ti (12GB VRAM) |
| 개발 환경 | macOS (Mac mini M4) — 코드 편집 / Git 관리 |
| 배포 형태 | ONNX → Mobile App (on-device) |
| 모델 사이즈 목표 | INT8 양자화 후 100~300MB |

---

## 빠른 시작 (학습 서버에서)

NVIDIA GPU가 있는 Linux 서버에서 아래 한 줄이면 환경이 준비됩니다.

```bash
# 1. 저장소 clone
git clone <THIS_REPO_URL> tts-finetune
cd tts-finetune

# 2. 자동 설치 (conda env + GPT-SoVITS clone + PyTorch cu128 + pretrained 모델)
bash setup.sh                       # 기본값: CU128 + HuggingFace
# 또는
bash setup.sh --device CU126 --source ModelScope --uvr5

# 3. 검증
conda activate gptsovits
python verify.py

# 4. 한국어 데이터셋 준비 (KSS 자동 다운로드 + GPT-SoVITS 형식 변환)
python scripts/prepare_kss.py --out-dir ~/data/kss_processed

# 5. 파인튜닝 (전처리 → s1 → s2)
source configs/experiment.env
python scripts/preprocess_for_training.py     # 30~60분
bash   scripts/train_s1.sh                    # GPT,    6~10시간 (tmux 권장)
bash   scripts/train_s2.sh                    # SoVITS, 8~14시간 (s1 끝난 뒤)
```

성공하면 모든 항목이 `[ OK ]` 로 표시됩니다. 자세한 단계·옵션·트러블슈팅은
[`docs/02_설치_가이드.md`](docs/02_설치_가이드.md) /
[`docs/03_데이터셋_준비.md`](docs/03_데이터셋_준비.md) /
[`docs/04_파인튜닝_가이드.md`](docs/04_파인튜닝_가이드.md) 를 참고하세요.

---

## 저장소 구조

```
.
├── README.md                       ← 본 문서 (저장소 메인)
├── LICENSE                         ← MIT
├── .gitignore                      ← 데이터셋·체크포인트·모델 가중치 등 제외
├── setup.sh                        ← 학습 서버용 자동 설치 스크립트
├── verify.py                       ← 설치 검증 스크립트
├── configs/
│   ├── experiment.env              ← 공통 환경변수 (경로/GPU/실험명)
│   ├── s1_kss.yaml                 ← GPT(s1) 학습 config (4070 Ti 12GB 튜닝)
│   └── s2_kss.json                 ← SoVITS(s2) 학습 config (4070 Ti 12GB 튜닝)
├── scripts/
│   ├── prepare_kss.py              ← KSS 다운로드 + GPT-SoVITS 학습 형식 변환
│   ├── preprocess_for_training.py  ← .list → BERT/HuBERT/semantic 일괄 추출
│   ├── train_s1.sh                 ← GPT(s1) 학습 launcher
│   └── train_s2.sh                 ← SoVITS(s2) 학습 launcher
└── docs/
    ├── 01_모델선정_비교.md          ← 후보 모델 비교 및 GPT-SoVITS v2 선정 사유
    ├── 02_설치_가이드.md            ← 단계별 설치·검증·트러블슈팅
    ├── 03_데이터셋_준비.md          ← KSS 다운로드·전처리·라이선스 가이드
    └── 04_파인튜닝_가이드.md        ← 전처리·s1·s2 학습·모니터링·트러블슈팅
```

> **주의:** `setup.sh` 실행 시 `~/projects/GPT-SoVITS` 에 GPT-SoVITS 본체가
> 별도로 clone 됩니다. 본 저장소에는 포함하지 않으며, `.gitignore` 에 제외 처리되어 있습니다.

---

## 워크플로 (Mac M4 ↔ 학습 서버)

```
┌─────────────────────────┐         git push          ┌──────────────────────────┐
│  Mac mini M4 (개발)      │ ───────────────────────▶ │  GitHub (origin)         │
│  - 코드 편집·문서 작성    │                          │  (public repo)           │
│  - 스크립트 / 설정 관리   │ ◀─────────────────────── │                          │
└─────────────────────────┘         git pull          └──────────────────────────┘
                                                                  │
                                                                  │ git clone / pull
                                                                  ▼
                                                       ┌──────────────────────────┐
                                                       │  Linux + RTX 4070 Ti     │
                                                       │  - bash setup.sh         │
                                                       │  - 데이터 전처리          │
                                                       │  - GPT/SoVITS 학습       │
                                                       │  - ONNX 변환             │
                                                       └──────────────────────────┘
```

- **Mac에서 하지 않는 것**: PyTorch CUDA 학습, 대용량 데이터 다운로드.
- **서버에 두지 않는 것**: 본 저장소가 아닌 임시 코드, 비밀키.
- **Git에 올리지 않는 것**: 오디오 데이터, 체크포인트, ONNX 가중치 (`.gitignore` 참고).

---

## 진행 단계 체크리스트

- [x] 0. 모델 선정 — GPT-SoVITS v2 ([01_모델선정_비교.md](docs/01_모델선정_비교.md))
- [x] 1. 학습 서버 자동 설치 스크립트 ([02_설치_가이드.md](docs/02_설치_가이드.md))
- [x] 2. 한국어 데이터셋 준비 — KSS 베이스라인 ([03_데이터셋_준비.md](docs/03_데이터셋_준비.md))
      *비상업 검증용. 상용 모델은 라이선스 호환 데이터셋으로 재학습.*
- [x] 3. GPT(s1) / SoVITS(s2) 파인튜닝 코드·config·가이드 ([04_파인튜닝_가이드.md](docs/04_파인튜닝_가이드.md))
      *서버에서 `bash scripts/train_s1.sh` / `train_s2.sh` 실행만 남음.*
- [ ] 5. 평가 (MOS, CER/WER, RTF)
- [ ] 6. ONNX export (s1.onnx, s2.onnx)
- [ ] 7. INT8 양자화 + ONNX Runtime Mobile 검증
- [ ] 8. Android / iOS 데모 앱 통합

---

## 라이선스

본 저장소 코드: **MIT License** ([LICENSE](LICENSE))

GPT-SoVITS 본체 및 의존 패키지는 각자의 라이선스를 따릅니다.
