#!/usr/bin/env bash
# =============================================================================
#  SoVITS (s2) 파인튜닝 launcher — KSS 베이스라인
#
#  사전 조건:
#    1. bash setup.sh
#    2. python scripts/prepare_kss.py
#    3. python scripts/preprocess_for_training.py
#  s1 학습과 독립적으로 실행 가능 (병렬 가능, 단 4070 Ti 12GB 단일이라면 순차 권장).
#
#  사용:
#    source configs/experiment.env
#    bash scripts/train_s2.sh
#
#  결과:
#    $EXP_DIR/logs_s2/         — TensorBoard + raw checkpoint (G_*.pth, D_*.pth)
#    $EXP_DIR/weights_s2/      — 추론용 가중치 (.pth)
# =============================================================================
set -Eeuo pipefail

: "${REPO_ROOT:?source configs/experiment.env first}"
: "${GPT_SOVITS_DIR:?source configs/experiment.env first}"
: "${EXP_DIR:?source configs/experiment.env first}"
: "${EXP_NAME:?source configs/experiment.env first}"
: "${S2_CONFIG:?source configs/experiment.env first}"
: "${PRETRAINED_S2G:?source configs/experiment.env first}"
: "${PRETRAINED_S2D:?source configs/experiment.env first}"
: "${GPU_ID:=0}"

S2_LOG_DIR="${EXP_DIR}/logs_s2"
S2_WEIGHT_DIR="${EXP_DIR}/weights_s2"
mkdir -p "${S2_LOG_DIR}" "${S2_WEIGHT_DIR}"

# ---------- 사전학습 가중치 점검 ----------
for f in "${PRETRAINED_S2G}" "${PRETRAINED_S2D}"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERR] pretrained s2 가 없습니다: $f"
    echo "      setup.sh 를 다시 실행하거나 GPT-SoVITS pretrained 다운로드를 확인하세요."
    exit 1
  fi
done

# ---------- 임시 config 생성 (사용자 json + 런타임 경로 주입) ----------
TMP_CONFIG="${EXP_DIR}/s2_runtime.json"
python - "$S2_CONFIG" "$TMP_CONFIG" "$EXP_DIR" "$EXP_NAME" "$S2_LOG_DIR" "$S2_WEIGHT_DIR" \
        "$PRETRAINED_S2G" "$PRETRAINED_S2D" "$GPU_ID" <<'PY'
import json, sys, pathlib
(src, dst, exp_dir, exp_name, log_dir, weight_dir,
 pretrained_g, pretrained_d, gpu_id) = sys.argv[1:]
cfg = json.loads(pathlib.Path(src).read_text(encoding="utf-8"))
# GPT-SoVITS s2 가 hps 로 읽는 핵심 필드
cfg["data"]["exp_dir"] = exp_dir
cfg["s2_ckpt_dir"] = log_dir
cfg["name"] = exp_name
cfg["save_weight_dir"] = weight_dir
cfg["train"]["pretrained_s2G"] = pretrained_g
cfg["train"]["pretrained_s2D"] = pretrained_d
cfg["train"]["gpu_numbers"] = gpu_id
pathlib.Path(dst).write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[INFO] runtime config 생성: {dst}")
PY

# ---------- 학습 실행 ----------
export _CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo
echo "========== SoVITS(s2) 파인튜닝 시작 =========="
echo "  exp_name      : ${EXP_NAME}"
echo "  config        : ${TMP_CONFIG}"
echo "  pretrained G  : ${PRETRAINED_S2G}"
echo "  pretrained D  : ${PRETRAINED_S2D}"
echo "  log dir       : ${S2_LOG_DIR}"
echo "  weight dir    : ${S2_WEIGHT_DIR}"
echo "  GPU           : ${GPU_ID}"
echo
echo "  TensorBoard   : tensorboard --logdir ${S2_LOG_DIR} --port 6007"
echo "============================================="
echo

cd "${GPT_SOVITS_DIR}"
exec python -s "GPT_SoVITS/s2_train.py" --config "${TMP_CONFIG}"
