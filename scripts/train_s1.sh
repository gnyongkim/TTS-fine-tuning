#!/usr/bin/env bash
# =============================================================================
#  GPT (s1) 파인튜닝 launcher — KSS 베이스라인
#
#  사전 조건:
#    1. bash setup.sh  (GPT-SoVITS 설치 + pretrained 다운로드)
#    2. python scripts/prepare_kss.py             (KSS → train.list/val.list)
#    3. python scripts/preprocess_for_training.py (train.list → BERT/HuBERT/semantic)
#
#  사용:
#    source configs/experiment.env
#    bash scripts/train_s1.sh
#
#  결과:
#    $EXP_DIR/logs_s1/         — TensorBoard + raw checkpoint (.ckpt)
#    $EXP_DIR/weights_s1/      — half-precision 가중치 (추론용 .ckpt)
# =============================================================================
set -Eeuo pipefail

# ---------- 환경변수 ----------
: "${REPO_ROOT:?source configs/experiment.env first}"
: "${GPT_SOVITS_DIR:?source configs/experiment.env first}"
: "${EXP_DIR:?source configs/experiment.env first}"
: "${EXP_NAME:?source configs/experiment.env first}"
: "${S1_CONFIG:?source configs/experiment.env first}"
: "${PRETRAINED_S1:?source configs/experiment.env first}"
: "${GPU_ID:=0}"
: "${IS_HALF:=True}"

S1_DIR="${EXP_DIR}"
LOG_DIR="${S1_DIR}/logs_s1"
WEIGHT_DIR="${S1_DIR}/weights_s1"
mkdir -p "${LOG_DIR}" "${WEIGHT_DIR}"

# ---------- 사전학습 가중치 점검 ----------
if [[ ! -f "${PRETRAINED_S1}" ]]; then
  echo "[ERR] pretrained s1 가 없습니다: ${PRETRAINED_S1}"
  echo "      setup.sh 를 다시 실행하거나 GPT-SoVITS pretrained 다운로드를 확인하세요."
  exit 1
fi

# ---------- 임시 config 생성 (사용자 yaml + 런타임 경로 주입) ----------
TMP_CONFIG="${S1_DIR}/s1_runtime.yaml"
python - "$S1_CONFIG" "$TMP_CONFIG" "$S1_DIR" "$EXP_NAME" "$WEIGHT_DIR" "$PRETRAINED_S1" <<'PY'
import sys, yaml, pathlib
src, dst, s1_dir, exp_name, weight_dir, pretrained = sys.argv[1:]
cfg = yaml.safe_load(pathlib.Path(src).read_text(encoding="utf-8"))
cfg.setdefault("train", {})
cfg["train"]["exp_name"] = exp_name
cfg["train"]["half_weights_save_dir"] = weight_dir
cfg["train"]["save_weight_dir"] = weight_dir
cfg["pretrained_s1"] = pretrained
cfg["train_semantic_path"] = f"{s1_dir}/6-name2semantic.tsv"
cfg["train_phoneme_path"]  = f"{s1_dir}/2-name2text.txt"
cfg["output_dir"] = f"{s1_dir}/logs_s1"
pathlib.Path(dst).write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
print(f"[INFO] runtime config 생성: {dst}")
PY

# ---------- 학습 실행 ----------
export _CUDA_VISIBLE_DEVICES="${GPU_ID}"
export hz="25hz"

echo
echo "========== GPT(s1) 파인튜닝 시작 =========="
echo "  exp_name      : ${EXP_NAME}"
echo "  config        : ${TMP_CONFIG}"
echo "  pretrained    : ${PRETRAINED_S1}"
echo "  log dir       : ${LOG_DIR}"
echo "  weight dir    : ${WEIGHT_DIR}"
echo "  GPU           : ${GPU_ID} (is_half=${IS_HALF})"
echo
echo "  TensorBoard   : tensorboard --logdir ${LOG_DIR} --port 6006"
echo "==========================================="
echo

cd "${GPT_SOVITS_DIR}"
exec python -s "GPT_SoVITS/s1_train.py" --config_file "${TMP_CONFIG}"
