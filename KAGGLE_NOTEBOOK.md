# Kaggle P100 노트북 실행 가이드 (MeloTTS KR 파인튜닝)

이 문서는 Kaggle 노트북 한 개를 열어, 아래 순서대로 셀을 복사-붙여넣기 하는 것만으로 MeloTTS 파인튜닝이 끝까지 돌아가도록 설계된 템플릿이다.

> 가정
> - Accelerator: **GPU P100** (16GB) — 비활성화돼 있으면 Kaggle 계정 설정 → Phone verification 먼저.
> - Internet: **On** — GitHub clone / HuggingFace 다운로드에 필요.
> - 데이터셋: `mozilla-foundation/common_voice_13_0` (Korean) 또는 이미 첨부된 Common Voice KR Dataset 둘 중 하나.
> - 세션 시간 제한: 한 세션 **최대 9시간**. 긴 학습은 여러 세션으로 쪼개고, 체크포인트를 Kaggle Dataset으로 보내 다음 세션에서 이어 받는다.

Kaggle 세션의 `/kaggle/working` 은 세션 종료 시 **휘발되지 않고 output으로 남지만 20GB 제한**이고, 노트북을 닫으면 다음 세션에서 바로 쓸 수 없다. 그래서 **중간 산출물(전처리 wav + metadata + checkpoint)은 반드시 Kaggle Dataset으로 저장**해서 이어 학습할 수 있게 한다. 마지막 셀에 그 절차를 넣어뒀다.

---

## Cell 1 — 환경 점검

```python
!nvidia-smi | head -n 20
import sys, torch, platform
print("python  :", sys.version.split()[0])
print("torch   :", torch.__version__, "cuda=", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu     :", torch.cuda.get_device_name(0))
    print("cap     :", torch.cuda.get_device_capability(0))
    print("vram GB :", torch.cuda.get_device_properties(0).total_memory / 1024**3)
print("arch    :", platform.machine())
```

기대 출력: `Tesla P100-PCIE-16GB`, `cuda=True`. 안 뜨면 Accelerator 설정 다시 확인.

---

## Cell 2 — 프로젝트 clone

```bash
%%bash
set -e
cd /kaggle/working
if [ ! -d melotts-fine-tuning ]; then
  git clone https://github.com/gnyongkim/melotts-fine-tuning.git
fi
cd melotts-fine-tuning
git pull --ff-only
echo "---"
ls -la
```

---

## Cell 3 — MeloTTS 본체 clone + editable install

```bash
%%bash
set -e
cd /kaggle/working/melotts-fine-tuning

# MeloTTS 본체 (아직 없으면)
if [ ! -d MeloTTS ]; then
  git clone https://github.com/myshell-ai/MeloTTS.git
fi

# 우리 requirements (librosa / soundfile / tqdm / jamo / g2pkk 등)
pip install -q -r requirements-extra.txt

# MeloTTS 본체 의존성 + editable install
pip install -q -e ./MeloTTS
# 한국어 텍스트 프론트엔드
pip install -q g2pkk jamo
# unidic-lite 는 일본어/중국어 쪽에서 import하는 경로 회피용 (가끔 없으면 ImportError)
python -m unidic download || true

echo "--- pip freeze (핵심만) ---"
pip list 2>/dev/null | egrep -i 'torch|melo|librosa|soundfile|g2p|jamo|tqdm' | head -n 30
```

> `pip install -e ./MeloTTS` 를 해야 `from melo.api import TTS` / `melo.preprocess_text` 가 파이썬 경로에서 잡힌다.

---

## Cell 4 — Common Voice 경로 지정 (사용자가 한 번 확인)

Kaggle 좌측 사이드바 **Add Data** → `mozilla-foundation/common_voice_13_0` 또는 이미 첨부한 한국어 CV 데이터셋을 누르고, 실제 경로가 어떻게 붙었는지 아래 셀로 확인한다.

```python
import glob, os

# 대부분 이런 경로 중 하나:
candidates = [
    "/kaggle/input/common-voice-korean",
    "/kaggle/input/mozilla-foundation-common-voice-13-0",
    "/kaggle/input/common-voice",
]
for c in candidates:
    if os.path.exists(c):
        print("FOUND:", c)

# TSV / clips 자동 탐지
tsvs = glob.glob("/kaggle/input/**/validated.tsv", recursive=True)
clips = glob.glob("/kaggle/input/**/clips", recursive=True)
print("validated.tsv 후보:")
for t in tsvs: print(" ", t)
print("clips 후보:")
for c in clips: print(" ", c)
```

찾은 경로 중 **Korean용** 하나를 아래 변수에 고정한다.

```python
CV_TSV = "/kaggle/input/common-voice-korean/ko/validated.tsv"   # ← 실제 경로로 교체
CV_CLIPS = "/kaggle/input/common-voice-korean/ko/clips"         # ← 실제 경로로 교체

assert os.path.exists(CV_TSV), CV_TSV
assert os.path.exists(CV_CLIPS), CV_CLIPS
print("OK")
```

---

## Cell 5 — (추천) 먼저 `--dry_run` 으로 필터 통계만 확인

어떤 화자를 쓸지, 투표 필터가 너무 빡센지 점검하는 단계. 오디오 변환은 하지 않으므로 10초 안에 끝난다.

```bash
%%bash -s "$CV_TSV" "$CV_CLIPS"
set -e
cd /kaggle/working/melotts-fine-tuning
python scripts/cv_to_metadata.py \
  --tsv "$1" \
  --clips_dir "$2" \
  --out_wav_dir /kaggle/working/processed_cv \
  --out /kaggle/working/metadata_cv.list \
  --top_n 1 \
  --min_up_votes 2 \
  --max_down_votes 0 \
  --min_chars 5 --max_chars 60 \
  --drop_questions --drop_casual \
  --min_dur 0.5 --max_dur 8.0 \
  --dry_run
```

출력에서 확인할 것
- **후보 (필터 통과)** 가 몇 건인지. 300건 미만이면 화자/투표 기준을 완화한다 (`--top_n 2` 또는 `--min_up_votes 1`).
- 상위 5~10 화자 목록의 녹음 수 편차. Top-1이 극단적으로 많으면 그대로, 아니면 `--top_n 2` 로 섞는다.

"Top-1만 쓰고 싶은데 숫자가 부족하다" 면 먼저 `--min_up_votes` 를 1로 내려보고, 그래도 부족하면 화자 세 명을 붙여 Multi-speaker로 간다. (MeloTTS는 원래 multi-speaker를 지원한다. 다만 알림 멘트용이면 **한 화자**가 일관성 측면에서 제일 좋다.)

---

## Cell 6 — 실제 전처리 (mp3 → wav 변환 + metadata 생성)

```bash
%%bash -s "$CV_TSV" "$CV_CLIPS"
set -e
cd /kaggle/working/melotts-fine-tuning
python scripts/cv_to_metadata.py \
  --tsv "$1" \
  --clips_dir "$2" \
  --out_wav_dir /kaggle/working/processed_cv \
  --out /kaggle/working/metadata_cv.list \
  --top_n 1 \
  --min_up_votes 2 \
  --max_down_votes 0 \
  --min_chars 5 --max_chars 60 \
  --drop_questions --drop_casual \
  --min_dur 0.5 --max_dur 8.0 \
  --target_sr 44100

echo ""
echo "=== 샘플 5줄 ==="
head -n 5 /kaggle/working/metadata_cv.list
echo ""
echo "총 라인 수: $(wc -l < /kaggle/working/metadata_cv.list)"
du -sh /kaggle/working/processed_cv
```

이 셀은 오디오 변환이 있어 **수십 분** 걸릴 수 있다 (TSV 크기와 필터 통과 수에 비례). 끝나면 `metadata_cv.list` 가 `절대경로|KR_CV|KR|텍스트` 포맷으로 작성돼 있어야 한다.

> **데이터 교체하려면?** KSS 쪽을 쓰고 싶으면 Cell 5~6을 `scripts/kss_to_metadata.py` 로 바꾸면 된다. 같은 포맷으로 `/kaggle/working/metadata_kss.list` 를 만들고, 이후 Cell 7에서 `--metadata` 만 갈아끼면 된다. Common Voice vs KSS 중 품질은 KSS가 보통 더 안정적이다 (단일 화자·스튜디오 녹음). 두 세트로 각각 학습해 비교해도 된다.

---

## Cell 7 — 사전학습 체크포인트 다운로드

```bash
%%bash
set -e
cd /kaggle/working/melotts-fine-tuning
python scripts/03_download_pretrained.py --out pretrained/KR
ls -la pretrained/KR
```

기대 파일: `checkpoint.pth` (G 제너레이터), `config.json`. 둘 다 있어야 Cell 9(학습)의 `--pretrain_G` 로 warm-start가 된다.

---

## Cell 8 — preprocess_text 실행 (MeloTTS 공식 전처리)

MeloTTS는 `melo/preprocess_text.py` 에서
  1) 텍스트 정규화 + cleaned 버전 저장
  2) train/val split
  3) **wav마다 BERT 임베딩 캐시(`.bert.pt`) 생성** — 이 단계가 느리다 (P100으로도 수백 문장이면 몇 분).
  4) `data/config.json` 에 speaker·symbol 정보 갱신
을 한 번에 수행한다.

```bash
%%bash
set -e
cd /kaggle/working/melotts-fine-tuning

# 1) base config를 data/config.json 으로 복사 (P100용)
cp configs/config_p100.json data/config.json

# 2) metadata 경로 맞춰 놓기 (04_preprocess.sh 가 data/metadata.list 기대)
cp /kaggle/working/metadata_cv.list data/metadata.list

# 3) 전처리 실행
cd MeloTTS/melo
python preprocess_text.py \
    --metadata /kaggle/working/melotts-fine-tuning/data/metadata.list \
    --config_path /kaggle/working/melotts-fine-tuning/data/config.json

echo "=== 생성 파일 ==="
ls -la /kaggle/working/melotts-fine-tuning/data/ | head -n 20
```

확인할 산출물
- `data/metadata.list.cleaned`
- `data/train.list`, `data/val.list`
- `data/config.json` (speaker·symbol 자동 갱신된 상태)
- 각 wav 옆에 `.bert.pt` (처음 한 번만 생성)

**오류가 나는 대표 케이스**
- `kykim/bert-kor-base` 다운로드 실패 → `pip install -U huggingface_hub` 후 재시도.
- `tokenizers` 버전 경고 → 대부분 무시 가능.
- OOM → preprocess 자체는 GPU를 많이 안 쓰지만, 혹시 나면 `CUDA_VISIBLE_DEVICES=""` 를 앞에 붙여 CPU로 돌려도 된다 (느리지만 확실).

---

## Cell 9 — 학습 시작

Kaggle 세션 9시간 한계 때문에 **에폭 대신 시간 기반으로 중단**하는 전략이 좋다. 여기서는 `timeout` 으로 8시간 30분 지나면 자동 종료시킨다 (Kaggle이 강제로 죽이기 전에 정상 종료).

```bash
%%bash
set -e
cd /kaggle/working/melotts-fine-tuning

export MODEL_NAME=ft_kr_cv
export CONFIG=/kaggle/working/melotts-fine-tuning/data/config.json
export PRETRAIN_G=/kaggle/working/melotts-fine-tuning/pretrained/KR/checkpoint.pth
export MASTER_PORT=10902
mkdir -p outputs/$MODEL_NAME

cd MeloTTS/melo

# 8시간 30분 후 SIGTERM 보내서 torchrun 정상 종료시키기.
# -s SIGTERM : MeloTTS train.py가 체크포인트 저장 루틴을 탈 수 있게.
timeout --signal=SIGTERM 8h30m \
  torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
    train.py \
    --c "$CONFIG" \
    --model "$MODEL_NAME" \
    --pretrain_G "$PRETRAIN_G" \
  2>&1 | tee /kaggle/working/melotts-fine-tuning/outputs/$MODEL_NAME/train.log
```

실행 중 확인 포인트 (tee한 로그에서)
- `total loss` 가 감소 추세인지.
- `mel loss` 가 40 언저리에서 점점 떨어지는지 (45 근처에서 고정되면 학습이 안 되는 중).
- **매 `eval_interval=500` 스텝마다** `logs/ft_kr_cv/G_<step>.pth` 가 쌓이는지.

---

## Cell 10 — 학습 중간 점검 (선택)

학습이 돌아가는 동안 다른 셀에서 최근 체크포인트만 꺼내서 샘플 합성.

```bash
%%bash
set -e
cd /kaggle/working/melotts-fine-tuning
LATEST=$(ls -1t MeloTTS/melo/logs/ft_kr_cv/G_*.pth 2>/dev/null | head -n 1)
if [ -z "$LATEST" ]; then
  echo "아직 체크포인트 없음"
  exit 0
fi
echo "latest: $LATEST"

python scripts/06_infer.py \
  --ckpt "$LATEST" \
  --config data/config.json \
  --out_dir outputs/ab_samples \
  --text "안녕하세요. 저는 오늘도 열심히 일하고 있습니다." \
         "새 알림이 도착했습니다." \
         "10분 뒤 회의가 시작됩니다."
ls -la outputs/ab_samples | tail -n 10
```

Kaggle 노트북에서 wav는 `IPython.display.Audio` 로 바로 들어볼 수 있다.

```python
from IPython.display import Audio
Audio("/kaggle/working/melotts-fine-tuning/outputs/ab_samples/0.wav")
```

---

## Cell 11 — (중요) 체크포인트·설정·전처리 결과를 Kaggle Dataset으로 내보내기

세션이 종료되면 `/kaggle/working` 은 남지만, 다음 세션에서 그대로 이어 돌리려면 Kaggle Dataset에 올려두는 게 편하다. 노트북 우측 상단 **Output → Save Version** 을 눌러도 되지만, 크기가 20GB를 넘기면 **Dataset으로 분리 저장**이 안전하다.

수동 절차
1. 우측 **Data** 패널 하단 **+ New Dataset**.
2. Source: `/kaggle/working/melotts-fine-tuning/MeloTTS/melo/logs/ft_kr_cv` 폴더 전체 압축 없이 업로드.
3. Dataset 이름: `melotts-ft-kr-cv-checkpoints` 같은 걸로.
4. 다음 세션에서 이 Dataset을 Add Data로 첨부하면 `/kaggle/input/melotts-ft-kr-cv-checkpoints/` 로 마운트된다.

> 이미 전처리에서 생성된 `.bert.pt` 들을 다시 만들기 싫으면, `processed_cv/` 폴더(wav + .bert.pt 쌍)를 **별도 Dataset**으로 한 번 올려두는 게 시간을 크게 아낀다.

이어 학습할 때는 Cell 9의 `--pretrain_G` 를 **Kaggle Dataset 경로의 최신 G_*.pth** 로 바꾸면 된다. 예:

```
--pretrain_G /kaggle/input/melotts-ft-kr-cv-checkpoints/G_12000.pth
```

---

## Cell 12 — 최종 체크포인트 내보내기 (랩업)

```bash
%%bash
set -e
cd /kaggle/working/melotts-fine-tuning
MODEL_NAME=ft_kr_cv
LOGS_DIR=MeloTTS/melo/logs/$MODEL_NAME
mkdir -p outputs/$MODEL_NAME

LATEST_G=$(ls -1t $LOGS_DIR/G_*.pth | head -n 1)
echo "latest G: $LATEST_G"
cp "$LATEST_G" outputs/$MODEL_NAME/G_latest.pth
cp data/config.json outputs/$MODEL_NAME/config.json

# 검증 샘플 wav 세 문장
python scripts/06_infer.py \
  --ckpt outputs/$MODEL_NAME/G_latest.pth \
  --config outputs/$MODEL_NAME/config.json \
  --out_dir outputs/$MODEL_NAME/samples \
  --text "안녕하세요. 오늘도 좋은 하루 되세요." \
         "지금 확인할 알림이 도착했습니다." \
         "다섯 시에 회의가 시작됩니다."

ls -la outputs/$MODEL_NAME
```

이 단계까지 오면 `outputs/ft_kr_cv/G_latest.pth` + `config.json` + `samples/*.wav` 묶음이 있다. 이걸 Kaggle Dataset으로 한 번 더 올려두면, 이후 ONNX 변환 단계에서 로컬로 내려받아 쓸 수 있다.

---

## 다음 단계 (Kaggle 이후)

1. **청취 평가**: `samples/*.wav` 를 알림 용도 시나리오에서 직접 듣고, 발음·리듬·피치가 자연스러운지 확인. 이상한 곳이 반복되면 그 음운 패턴(예: 종성 'ㄹ' 연음, 긴 복합어)을 포함한 문장을 metadata에 추가해 소량 재학습.
2. **ONNX 변환은 로컬(Mac)에서**: Kaggle에서 받은 `G_latest.pth` + `config.json` 을 `/Users/kimguenyong/Documents/workspace/study/melotts-export/` 쪽에 복사한 뒤 기존 `02_convert_onnx.py` → `03_quantize.py` → `07_export_no_bert.py` 순서로 수행. Kaggle에서 ONNX까지 돌릴 필요는 없다 (세션 시간 아깝다).
3. **모바일 번들**: `MOBILE_DEPLOYMENT.md` 의 Phase B (g2pkk 포팅) + Phase C (런타임 통합) 로 이어진다.

---

## 자주 만나는 문제

| 증상 | 원인 | 해결 |
|---|---|---|
| `torch.cuda.OutOfMemoryError` | `batch_size` 너무 큼 | `configs/config_p100.json` 에서 `batch_size` 를 6 → 4 → 2 로 내림. `segment_size` 도 16384 → 8192. |
| preprocess에서 BERT 다운로드 실패 | HF 토큰/네트워크 | `!huggingface-cli login` 이후 재시도. 사실상 public 모델이라 토큰 없이 되는 게 정상. |
| loss가 NaN 으로 튐 | fp16 + 데이터 이상치 | `fp16_run: false` 로 내려 안정화, 또는 `cv_to_metadata.py` 에서 `--max_dur` 를 더 짧게 (긴 발화 튀는 샘플이 원인인 경우 많음). |
| eval 단계에서 오래 멈춤 | `eval_interval` 이 너무 잦음 | `config_p100.json` 에서 500 → 1000. |
| `torchrun` 이 바로 종료 | `--pretrain_G` 경로 오타 | `ls pretrained/KR/checkpoint.pth` 로 먼저 확인. |
| 세션이 9시간 전에 죽음 | 유휴 감지 | 다른 셀에서 `while True: print('.'); time.sleep(60)` 같은 키핑 셀은 권장 안 함 (Kaggle 정책 위반 가능). 대신 `timeout 8h30m` 으로 안전 종료 후 Dataset 저장 루틴을 확실하게. |
