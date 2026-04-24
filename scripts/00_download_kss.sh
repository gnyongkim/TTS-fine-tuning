#!/usr/bin/env bash
# KSS (Korean Single Speaker Speech) 데이터셋을 Kaggle CLI로 다운로드.
#
# 사전 준비:
#   1) pip install kaggle
#   2) Kaggle API 토큰 발급:
#        https://www.kaggle.com/settings/account  →  "Create New API Token"
#        다운받은 kaggle.json 을 ~/.kaggle/kaggle.json 에 저장하고 권한 조정:
#        mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
#
# 다운로드되는 것 (한 덩어리):
#   - transcript.v.1.4.txt
#   - kss/{1,2,3,4}/*.wav   (약 12,854개, 총 ~4-5GB, 원본 44.1kHz)

set -e

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

TARGET_DIR="${TARGET_DIR:-data/kss}"
mkdir -p "$TARGET_DIR"

# kaggle CLI 존재 확인
if ! command -v kaggle &> /dev/null; then
  echo "!! 'kaggle' CLI가 설치되어 있지 않습니다."
  echo "   pip install kaggle"
  exit 1
fi

# API 토큰 존재 확인
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "!! Kaggle API 토큰이 없습니다: $HOME/.kaggle/kaggle.json"
  echo "   1) https://www.kaggle.com/settings/account 에서 'Create New API Token' 클릭"
  echo "   2) 다운받은 kaggle.json 을 다음 위치로 이동:"
  echo "      mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/"
  echo "      chmod 600 ~/.kaggle/kaggle.json"
  exit 1
fi

echo "=============================================="
echo " KSS 데이터셋 다운로드 시작"
echo "   대상: $TARGET_DIR"
echo "   용량: 약 4-5GB (압축), 압축 해제 후 ~8GB"
echo "=============================================="

# 다운로드 + 자동 압축 해제 (-p: 저장 경로, --unzip: 자동 해제)
kaggle datasets download \
  -d bryanpark/korean-single-speaker-speech-dataset \
  -p "$TARGET_DIR" \
  --unzip

echo ""
echo "[verify] 받은 구조 확인..."
if [ -f "$TARGET_DIR/transcript.v.1.4.txt" ]; then
  N_LINES=$(wc -l < "$TARGET_DIR/transcript.v.1.4.txt")
  echo "  [OK] transcript.v.1.4.txt ($N_LINES 라인)"
else
  echo "  [??] transcript.v.1.4.txt 가 보이지 않습니다."
fi

if [ -d "$TARGET_DIR/kss" ]; then
  N_WAV=$(find "$TARGET_DIR/kss" -name '*.wav' | wc -l)
  echo "  [OK] kss/ 디렉토리 ($N_WAV 개 wav)"
else
  echo "  [??] kss/ 디렉토리가 보이지 않습니다. 압축 구조가 다를 수 있습니다."
fi

echo ""
echo "[DONE] 다음 단계:"
echo "  python scripts/kss_to_metadata.py \\"
echo "    --kss_dir    $TARGET_DIR/kss \\"
echo "    --transcript $TARGET_DIR/transcript.v.1.4.txt \\"
echo "    --out        data/metadata.list \\"
echo "    --speaker KR_KSS --max_dur 3.0 --max_chars 30 --drop_questions"
