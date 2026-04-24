# MeloTTS 한국어 파인튜닝 프로젝트

**목표**: [MeloTTS-Korean](https://huggingface.co/myshell-ai/MeloTTS-Korean)을 **알림/안내 멘트(짧은 문장)** 도메인에 맞춰 더 자연스럽게 파인튜닝.

**환경 전제**: RTX 4060 (VRAM 8GB), Ubuntu/Windows(WSL2) + CUDA 12.x, Python 3.10.

---

## ⚠️ 먼저 알아둘 점 (매우 중요)

"MeloTTS **기본 화자 목소리 그대로** 유지하면서 알림 도메인만 개선"은 TTS 파인튜닝에서 가장 까다로운 목표입니다. 이유:

1. 파인튜닝은 "오디오+텍스트 쌍" 데이터로 이루어집니다.
2. MeloTTS 기본 KR 화자와 **똑같은 목소리의 추가 녹음**은 공개되지 않았습니다.
3. 따라서 실제로 할 수 있는 현실적 옵션은:

| 옵션 | 방법 | 목소리 | 난이도 |
|------|------|--------|-------|
| A | 기본 화자 톤과 비슷한 성우로 300~1000문장 녹음 후 파인튜닝 | 소폭 변경 | ★★★ |
| B | [KSS 데이터셋](https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset)에서 알림 스타일 문장만 선별해 파인튜닝 | 바뀜(KSS 여자 화자) | ★★ |
| C | 파인튜닝 없이 **텍스트 정규화 + 프롬프트 튜닝**만 개선 | 유지 | ★ |

이 프로젝트는 **옵션 A 또는 B**를 기본 전제로 스크립트를 제공합니다. 옵션 C만 원하시면 `scripts/06_infer.py`의 `text_normalize()` 함수만 커스터마이징하면 됩니다.

---

## 전체 흐름 (9단계)

```
1. 환경 셋업           → setup_env.sh
2. 환경 검증           → scripts/verify_env.py
3. 녹음 스크립트 준비  → data/scripts/notification_scripts.txt
4. 녹음 (또는 데이터셋 다운로드)
5. 오디오 전처리       → scripts/01_prepare_audio.py
6. 메타데이터 구축     → scripts/02_build_metadata.py
7. 사전학습 모델 받기  → scripts/03_download_pretrained.py
8. MeloTTS 전처리      → scripts/04_preprocess.sh
9. 파인튜닝            → scripts/05_train.sh
10. 품질 검증 (A/B)    → scripts/06_infer.py
```

---

## 1. 환경 셋업

```bash
# 프로젝트 루트에서
bash setup_env.sh
```

이 스크립트는:
- `MeloTTS` 레포지토리를 프로젝트 하위로 clone
- conda/venv 생성은 직접 하시고, 그 안에서 `pip install -e MeloTTS/`
- 한국어 의존성(`g2pkk`, `jamo`, `mecab-ko-dic`) 설치
- `unidic` 다운로드

**수동 대안** (권장):
```bash
conda create -n melotts python=3.10 -y
conda activate melotts
pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/myshell-ai/MeloTTS.git
cd MeloTTS && pip install -e . && cd ..
python -m unidic download
pip install -r requirements-extra.txt
```

## 2. 환경 검증

```bash
python scripts/verify_env.py
```

GPU / VRAM / PyTorch CUDA / MeloTTS import / 한국어 g2p 동작을 차례로 확인합니다.

## 3. 녹음 스크립트 준비

`data/scripts/notification_scripts.txt`에 400+ 개의 알림/안내 멘트 예시를 포함했습니다. 길이는 대부분 1~3초 분량입니다.

필요하면 `data/scripts/custom.txt`를 만들어 본인의 서비스에 맞는 문구를 추가하세요.

## 4. 녹음 (옵션 A) 또는 KSS 활용 (옵션 B)

### 옵션 A — 직접 녹음
- 조용한 환경, 한 성우가 같은 톤과 속도로 일관되게.
- **최소 300개, 권장 1000개 이상**.
- 포맷: 모노 WAV, 24kHz~48kHz(무엇이든 가능, 나중에 리샘플링함).
- 파일명: `0001.wav ~ NNNN.wav` — 텍스트 파일의 N번째 라인과 매칭.
- `data/raw/` 밑에 저장.

### 옵션 B — KSS 서브셋 (업로드하신 노트북이 쓰던 그 데이터셋)

참고: 업로드하신 `fine-tune-whisper-korean.ipynb`는 **Whisper(ASR)** 파인튜닝 노트북이라
MeloTTS 학습 코드로는 **직접 쓸 수 없습니다** (방향과 아키텍처가 완전히 다름).
다만 그 노트북이 쓰는 **KSS 데이터셋은 옵션 B의 재료로 그대로 활용 가능합니다**.

절차:

```bash
# 1) Kaggle API 토큰 세팅 (최초 1회):
#    https://www.kaggle.com/settings/account → "Create New API Token"
#    pip install kaggle
#    mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# 1b) KSS 데이터셋 전체 다운로드 (~4-5GB; transcript + wav 12,854개 한 번에)
bash scripts/00_download_kss.sh
# → data/kss/transcript.v.1.4.txt + data/kss/kss/{1,2,3,4}/*.wav 구조로 받아짐

# 2) MeloTTS metadata.list 변환 + 짧은 알림 스타일 필터
python scripts/kss_to_metadata.py \
  --kss_dir    data/kss \
  --transcript data/kss/transcript.v.1.4.txt \
  --out        data/metadata.list \
  --speaker    KR_KSS \
  --max_dur 3.0 --min_dur 0.5 --max_chars 30 --drop_questions

# 3) KSS 샘플레이트가 44.1kHz가 아니면 리샘플링 (data/processed로)
#    KSS 원본은 44.1kHz 16bit이지만, 커뮤니티 미러 중에는 16kHz 버전도 있음.
#    확인: python -c "import soundfile as sf; print(sf.info('data/kss/1/1_0000.wav'))"
#    44.1kHz가 아니면:
python scripts/01_prepare_audio.py --in_dir data/kss --out_dir data/processed --target_sr 44100
# 그 뒤 metadata.list의 경로를 data/processed/ 로 바꿔서 다시 빌드하거나,
# 02_build_metadata.py로 data/processed 기준 새 metadata를 만드세요.
```

필터링 팁 (`kss_to_metadata.py`):
- `--max_dur 3.0 --max_chars 30` → **알림 멘트 길이**에 맞춤 (가장 추천)
- `--drop_questions` → 의문문 제외 (알림은 평서/명령이 대부분)
- `--limit 1000` → 빠른 테스트용으로 1000문장만

**주의**: config의 `spk2id`를 `{"KR_KSS": 0}` 으로 맞추세요. 본 저장소의 `configs/config_4060.json`에서 `spk2id` 값을 `{ "KR_KSS": 0 }` 으로 변경한 뒤 `bash scripts/04_preprocess.sh` 로 진행하면 됩니다.

## 5. 오디오 전처리

```bash
python scripts/01_prepare_audio.py \
  --in_dir  data/raw \
  --out_dir data/processed \
  --target_sr 44100
```

작업: 44.1kHz 리샘플링, 모노 변환, 시작/끝 무음 trim, 피크 정규화 -1dB.

## 6. 메타데이터 구축

```bash
python scripts/02_build_metadata.py \
  --wav_dir data/processed \
  --script  data/scripts/notification_scripts.txt \
  --speaker KR_FT \
  --lang    KR \
  --out     data/metadata.list
```

출력 포맷 (MeloTTS 표준):
```
data/processed/0001.wav|KR_FT|KR|안녕하세요. 주문이 완료되었습니다.
data/processed/0002.wav|KR_FT|KR|배송이 시작되었습니다.
...
```

## 7. 사전학습 모델 다운로드

```bash
python scripts/03_download_pretrained.py --out pretrained/KR
```

HuggingFace Hub에서 `G.pth`, `D.pth`, `DUR.pth` 및 `config.json` 다운로드.

## 8. MeloTTS 전처리 (phones/tones/word2ph + BERT 임베딩)

```bash
bash scripts/04_preprocess.sh
```

내부적으로 `python MeloTTS/melo/preprocess_text.py --metadata <abs>/data/metadata.list --config_path <abs>/data/config.json` 실행.
- `configs/config_4060.json` 을 `data/config.json` 으로 복사해 base config로 사용
- 텍스트 → 자모 phonemes 변환 (`g2pkk` 사용)
- BERT 임베딩을 `.bert.pt`로 사전 캐싱 (학습 중 BERT 재실행 안 함 → VRAM 절약)
- `data/train.list` / `data/val.list` 자동 분할
- `data/config.json` 이 실제 스피커/심볼 정보로 갱신됨 (이후 단계에서 이 파일을 사용)

## 9. 파인튜닝 실행

```bash
bash scripts/05_train.sh
```

- `data/config.json` (전처리에서 갱신된 것) 사용 — batch_size=2, fp16_run=true, segment_size=8192
- 사전학습 `checkpoint.pth`를 `--pretrain_G`로 로드해 **warm-start**
  - (MeloTTS-Korean은 D/DUR 체크포인트를 공개하지 않아 이 둘은 랜덤 초기화. 초기 몇 epoch은 GAN 품질이 잠깐 떨어졌다 회복되는 패턴이 정상입니다.)
- `torchrun --nproc_per_node=1`로 단일 GPU 학습
- 체크포인트는 `MeloTTS/melo/logs/ft_kr/` 에 저장되며, 학습 후 `outputs/ft_kr/G_latest.pth` 와 `outputs/ft_kr/config.json` 으로 복사됨
- tensorboard: `tensorboard --logdir MeloTTS/melo/logs/ft_kr`

**예상 시간**: RTX 4060에서 1000문장·30 epoch 기준 6~12시간.

**VRAM 관리 팁**:
- OOM 발생 시 `config_4060.json`에서 `batch_size`를 1로, `segment_size`를 4096으로 줄이세요.
- Windows에서는 워커(`num_workers`)를 2 이하로 두세요.

## 10. 품질 검증 (A/B)

```bash
python scripts/06_infer.py \
  --ckpt           outputs/ft_kr/G_latest.pth \
  --config         outputs/ft_kr/config.json \
  --baseline_ckpt  pretrained/KR/checkpoint.pth \
  --baseline_config pretrained/KR/config.json \
  --out_dir        outputs/ab_samples
```

동일 문장 리스트(`data/scripts/eval_sentences.txt`)를 **기본 모델**과 **파인튜닝 모델**로 합성해 `outputs/ab_samples/` 아래 나란히 저장. MOS 블라인드 비교를 위해 랜덤 순서 매핑도 csv로 저장.

---

## 트러블슈팅

| 증상 | 원인 / 해결 |
|------|------------|
| `CUDA out of memory` | batch_size 1, segment_size 4096, num_workers 0으로 |
| `g2pkk` import 에러 | `pip install g2pkk jamo` |
| `MeCab` 관련 에러 | `pip install mecab-python3 unidic-lite`; 또는 `apt install mecab libmecab-dev mecab-ipadic-utf8` |
| Windows에서 학습 멈춤 | WSL2 권장. 네이티브 Windows는 `num_workers=0` 필수 |
| 학습 loss가 발산 | lr을 `0.0001`로 낮추고, `warmup_epochs`를 늘려보세요 |
| 한글이 영어로 읽힘 | filelist의 language 코드가 `KR`인지 확인 |

## 관련 문서

- [`GIT_SETUP.md`](./GIT_SETUP.md) — Git 초기화 · GitHub 업로드 · 학습 PC에서 clone 하는 루틴
- [`MOBILE_DEPLOYMENT.md`](./MOBILE_DEPLOYMENT.md) — 학습 이후 ONNX 변환 + 모바일 on-device 배포 로드맵 (옵션 4)

## 참고

- MeloTTS 공식: https://github.com/myshell-ai/MeloTTS
- KR 모델: https://huggingface.co/myshell-ai/MeloTTS-Korean
- VITS 원 논문: https://arxiv.org/abs/2106.06103
