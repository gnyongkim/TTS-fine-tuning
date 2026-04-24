#!/usr/bin/env bash
# 파인튜닝 런처 (RTX 4060 / 8GB VRAM 가정).
# MeloTTS train.py를 torchrun으로 직접 호출하고 --pretrain_G 플래그로 warm-start.
# 학습 체크포인트는 MeloTTS/melo/logs/<MODEL_NAME>/ 아래에 저장되고,
# 끝나면 outputs/<MODEL_NAME>/ 으로 최신본을 복사합니다.

set -e

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

# 학습에 쓸 config: preprocess가 업데이트한 data/config.json 을 사용
CONFIG="${CONFIG:-$PROJECT_ROOT/data/config.json}"
PRETRAIN_G="${PRETRAIN_G:-$PROJECT_ROOT/pretrained/KR/checkpoint.pth}"
MODEL_NAME="${MODEL_NAME:-ft_kr}"
MASTER_PORT="${MASTER_PORT:-10902}"

if [ ! -f "$CONFIG" ]; then
  echo "!! config 없음: $CONFIG"
  echo "   먼저 bash scripts/04_preprocess.sh 를 실행하세요."
  exit 1
fi
if [ ! -f "$PRETRAIN_G" ]; then
  echo "!! 사전학습 G 체크포인트 없음: $PRETRAIN_G"
  echo "   python scripts/03_download_pretrained.py --out pretrained/KR 을 먼저 실행하세요."
  exit 1
fi

mkdir -p "$PROJECT_ROOT/outputs/$MODEL_NAME"

echo "======================================================"
echo " MeloTTS KR Fine-tuning"
echo "   config      : $CONFIG"
echo "   pretrain_G  : $PRETRAIN_G"
echo "   model_name  : $MODEL_NAME"
echo "   master_port : $MASTER_PORT"
echo "======================================================"

cd MeloTTS/melo

# 단일 GPU (RTX 4060) → nproc_per_node=1.
# pretrain_D / pretrain_dur는 HuggingFace에 공개돼 있지 않으면 생략 가능.
torchrun --nproc_per_node=1 --master_port="$MASTER_PORT" \
    train.py \
    --c "$CONFIG" \
    --model "$MODEL_NAME" \
    --pretrain_G "$PRETRAIN_G" \
    2>&1 | tee "$PROJECT_ROOT/outputs/$MODEL_NAME/train.log"

# 학습 끝나면 최신 G 체크포인트를 outputs/로 복사
cd "$PROJECT_ROOT"
LOGS_DIR="MeloTTS/melo/logs/$MODEL_NAME"
if [ -d "$LOGS_DIR" ]; then
  LATEST_G=$(ls -1t "$LOGS_DIR"/G_*.pth 2>/dev/null | head -n 1)
  if [ -n "$LATEST_G" ]; then
    echo ""
    echo "[OK] 최신 체크포인트: $LATEST_G"
    cp "$LATEST_G" "outputs/$MODEL_NAME/G_latest.pth"
    cp "$CONFIG" "outputs/$MODEL_NAME/config.json"
    echo "     → outputs/$MODEL_NAME/G_latest.pth"
    echo "     → outputs/$MODEL_NAME/config.json"
  fi
fi

echo ""
echo "[DONE] 다음 단계: python scripts/06_infer.py --ckpt outputs/$MODEL_NAME/G_latest.pth \\"
echo "                    --config outputs/$MODEL_NAME/config.json \\"
echo "                    --out_dir outputs/ab_samples"
