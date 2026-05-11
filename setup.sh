#!/usr/bin/env bash
# =============================================================================
#  GPT-SoVITS v2 자동 설치 스크립트
#  - 대상: NVIDIA RTX 4070 Ti (CUDA 12.x), Linux (Ubuntu 22.04 LTS 권장)
#  - Python 3.10 / PyTorch 2.x (cu128) / GPT-SoVITS 공식 install.sh wrapper
#
#  사용:
#    bash setup.sh                       # 기본값(권장): CU128 + HuggingFace
#    bash setup.sh --device CU126        # CUDA 12.6 wheel 사용
#    bash setup.sh --source ModelScope   # HF가 막힐 때 ModelScope에서 다운로드
#    bash setup.sh --uvr5                # UVR5(보컬 분리) 모델까지 다운로드
#    bash setup.sh --install-dir ~/work  # 설치 위치 변경 (기본: ~/projects)
# =============================================================================
set -Eeuo pipefail

# ---------- 기본 설정 -----------------------------------------------------
ENV_NAME="gptsovits"
PY_VER="3.10"
INSTALL_DIR="${HOME}/projects"
REPO_URL="https://github.com/RVC-Boss/GPT-SoVITS.git"
DEVICE="CU128"   # CU126 | CU128
SOURCE="HF"      # HF | HF-Mirror | ModelScope
DOWNLOAD_UVR5="false"

# ---------- 색상 ---------------------------------------------------------
C_RESET="\033[0m"; C_INFO="\033[1;32m"; C_WARN="\033[1;33m"; C_ERR="\033[1;31m"; C_OK="\033[1;34m"
log()  { echo -e "${C_INFO}[INFO]${C_RESET}  $*"; }
warn() { echo -e "${C_WARN}[WARN]${C_RESET}  $*"; }
err()  { echo -e "${C_ERR}[ERR ]${C_RESET}  $*" >&2; }
ok()   { echo -e "${C_OK}[ OK ]${C_RESET}  $*"; }

# ---------- 인자 파싱 -----------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)       DEVICE="$2"; shift 2 ;;
    --source)       SOURCE="$2"; shift 2 ;;
    --uvr5)         DOWNLOAD_UVR5="true"; shift ;;
    --install-dir)  INSTALL_DIR="$2"; shift 2 ;;
    --env-name)     ENV_NAME="$2"; shift 2 ;;
    -h|--help)
      grep -E "^# " "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) err "Unknown argument: $1"; exit 1 ;;
  esac
done

REPO_DIR="${INSTALL_DIR}/GPT-SoVITS"

# =========================================================================
# 0. 사전 점검
# =========================================================================
log "========== [0/5] 사전 점검 =========="

# 0-1. OS 체크
if [[ "$(uname -s)" != "Linux" ]]; then
  err "이 스크립트는 Linux 전용입니다. (현재: $(uname -s))"; exit 1
fi
ok "OS: $(uname -s) $(uname -r) $(uname -m)"

# 0-2. NVIDIA 드라이버 / GPU
if ! command -v nvidia-smi >/dev/null 2>&1; then
  err "nvidia-smi 명령을 찾을 수 없습니다. NVIDIA 드라이버가 설치되어 있는지 확인하세요."; exit 1
fi
GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -n1)
ok "GPU: ${GPU_INFO}"
if ! echo "$GPU_INFO" | grep -qi "4070"; then
  warn "RTX 4070 Ti가 아닐 수 있습니다. 계속 진행은 가능합니다."
fi

# 0-3. CUDA 드라이버 버전 (Driver가 12.x 이상이면 cu128/cu126 둘 다 OK)
DRV_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | cut -d. -f1)
if [[ "$DRV_VER" -lt 525 ]]; then
  warn "NVIDIA 드라이버 ${DRV_VER}은(는) CUDA 12.x에 부족할 수 있습니다. (>=535 권장)"
fi

# 0-4. 디스크 공간 (최소 30GB 권장: pretrained + venv + 데이터)
FREE_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
if [[ "$FREE_GB" -lt 30 ]]; then
  warn "홈 디렉토리 가용 공간이 ${FREE_GB}GB 입니다. 최소 30GB 권장."
fi
ok "Free disk in \$HOME: ${FREE_GB}GB"

# 0-5. conda 체크
if ! command -v conda >/dev/null 2>&1; then
  err "conda를 찾을 수 없습니다. Miniconda를 먼저 설치하세요."
  echo
  echo "  설치 예시:"
  echo "    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh"
  echo "    bash /tmp/mc.sh -b -p \$HOME/miniconda3"
  echo "    \$HOME/miniconda3/bin/conda init bash && exec bash"
  exit 1
fi
ok "conda: $(conda --version)"

# 0-6. git, wget, unzip
for bin in git wget unzip; do
  command -v "$bin" >/dev/null 2>&1 || { err "필수 명령 '$bin' 가 없습니다."; exit 1; }
done

# =========================================================================
# 1. conda 환경 생성/활성화
# =========================================================================
log "========== [1/5] conda 환경 (${ENV_NAME}, Python ${PY_VER}) =========="

# conda를 현재 셸에서 사용할 수 있도록 초기화
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  warn "환경 '${ENV_NAME}' 이 이미 존재합니다. 그대로 사용합니다."
else
  log "새 환경 생성: ${ENV_NAME}"
  conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
fi
conda activate "${ENV_NAME}"
ok "Active env: $(conda info --envs | awk '/\*/ {print $1}')"
ok "Python: $(python --version)"

# pip 최신화
python -m pip install -q --upgrade pip wheel setuptools

# =========================================================================
# 2. GPT-SoVITS 저장소 clone
# =========================================================================
log "========== [2/5] GPT-SoVITS 저장소 clone =========="
mkdir -p "${INSTALL_DIR}"
if [[ -d "${REPO_DIR}/.git" ]]; then
  warn "이미 clone 되어 있습니다: ${REPO_DIR} (git pull 시도)"
  git -C "${REPO_DIR}" pull --ff-only || warn "git pull 실패 — 기존 코드 그대로 진행합니다."
else
  git clone --depth 1 "${REPO_URL}" "${REPO_DIR}"
fi
ok "Repo: ${REPO_DIR}"

# =========================================================================
# 3. 공식 install.sh 실행 (PyTorch + 의존성 + pretrained 다운로드)
# =========================================================================
log "========== [3/5] 공식 install.sh 실행 (device=${DEVICE}, source=${SOURCE}) =========="
cd "${REPO_DIR}"

INSTALL_ARGS=(--device "${DEVICE}" --source "${SOURCE}")
if [[ "${DOWNLOAD_UVR5}" == "true" ]]; then
  INSTALL_ARGS+=(--download-uvr5)
fi
log "→ bash install.sh ${INSTALL_ARGS[*]}"
bash install.sh "${INSTALL_ARGS[@]}"

# =========================================================================
# 4. 한국어 추가 점검 (mecab 사전, g2pk2)
# =========================================================================
log "========== [4/5] 한국어 의존성 점검 =========="

python - <<'PY' || true
import importlib, sys
mods = ["torch", "torchaudio", "transformers", "g2pk2", "ko_pron", "python_mecab_ko"]
missing = []
for m in mods:
    try:
        importlib.import_module(m if m != "python_mecab_ko" else "mecab")
        print(f"  [OK] {m}")
    except Exception as e:
        print(f"  [MISS] {m}: {e}")
        missing.append(m)
if missing:
    print("\n[!] 누락된 모듈:", missing)
PY

# =========================================================================
# 5. CUDA / PyTorch 빠른 검증
# =========================================================================
log "========== [5/5] CUDA / PyTorch 검증 =========="
python - <<'PY'
import torch, sys
print("Python  :", sys.version.split()[0])
print("PyTorch :", torch.__version__)
print("CUDA OK :", torch.cuda.is_available())
print("CUDA Ver:", torch.version.cuda)
print("Device  :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("Capable :", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "N/A")
print("VRAM(GB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
      if torch.cuda.is_available() else "N/A")
PY

ok "설치 완료!"
echo
echo "=========================================================================="
echo "  다음 단계 (WebUI 실행)"
echo "    conda activate ${ENV_NAME}"
echo "    cd ${REPO_DIR}"
echo "    python webui.py ko_KR              # 한국어 UI로 실행"
echo "    # → http://localhost:9874 접속"
echo
echo "  검증 스크립트 (선택):"
echo "    python ${INSTALL_DIR}/verify.py"
echo "=========================================================================="
