#!/usr/bin/env bash
# MeloTTS 텍스트/BERT 전처리.
# 주의: MeloTTS의 preprocess_text.py는 다음 동작을 합니다.
#   - 입력:  --metadata <path>  (기본: data/example/metadata.list)
#           --config_path <base config>  (기본: configs/config.json)
#   - 출력:  {metadata_dir}/metadata.list.cleaned
#           {metadata_dir}/train.list
#           {metadata_dir}/val.list
#           {metadata_dir}/config.json   ← 우리 base를 기반으로 speaker/symbol 갱신
#           각 wav 파일 옆에 .bert.pt (BERT 임베딩 사전 캐시)
#
# 따라서 이 스크립트는:
#   1) configs/config_4060.json 을 data/config.json 으로 복사 (base로 사용)
#   2) MeloTTS/melo 디렉토리에서 preprocess_text.py 실행
#   3) 생성된 data/config.json 을 다음 단계(학습)에서 사용

set -e

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

BASE_CONFIG="${BASE_CONFIG:-$PROJECT_ROOT/configs/config_4060.json}"
METADATA="${METADATA:-$PROJECT_ROOT/data/metadata.list}"
RUN_CONFIG="$PROJECT_ROOT/data/config.json"

if [ ! -f "$METADATA" ]; then
  echo "!! metadata.list 없음: $METADATA"
  echo "   먼저 scripts/02_build_metadata.py 를 실행하세요."
  exit 1
fi
if [ ! -f "$BASE_CONFIG" ]; then
  echo "!! base config 없음: $BASE_CONFIG"; exit 1
fi
if [ ! -d "MeloTTS" ]; then
  echo "!! MeloTTS 디렉토리 없음. bash setup_env.sh 를 먼저 실행하세요."
  exit 1
fi

echo "[1/3] base config 복사"
echo "       $BASE_CONFIG  →  $RUN_CONFIG"
cp "$BASE_CONFIG" "$RUN_CONFIG"

echo ""
echo "[2/3] MeloTTS preprocess_text 실행"
echo "       metadata  = $METADATA"
echo "       config    = $RUN_CONFIG  (in-place 업데이트)"
cd MeloTTS/melo
python preprocess_text.py \
    --metadata "$METADATA" \
    --config_path "$RUN_CONFIG"
cd "$PROJECT_ROOT"

echo ""
echo "[3/3] 생성 파일 확인"
for f in "$METADATA.cleaned" "$PROJECT_ROOT/data/train.list" "$PROJECT_ROOT/data/val.list" "$RUN_CONFIG"; do
  if [ -f "$f" ]; then
    echo "   [OK] $f"
  else
    echo "   [??] $f  (생성되지 않음 — preprocess_text.py 출력 확인 필요)"
  fi
done

echo ""
echo "[DONE] 전처리 완료. 다음 단계: bash scripts/05_train.sh"
