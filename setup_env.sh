#!/usr/bin/env bash
# MeloTTS 한국어 파인튜닝 환경 셋업 스크립트
# 전제: conda 또는 venv 가상환경이 이미 활성화되어 있어야 함.
#      (예: conda create -n melotts python=3.10 -y && conda activate melotts)

set -e

echo "=============================================="
echo " MeloTTS Korean fine-tuning environment setup"
echo "=============================================="

# 0. 현재 Python 확인
python --version
which python
which pip

# 1. PyTorch (CUDA 12.1 빌드) — RTX 4060은 CUDA 12.x 필요
echo "[1/5] Installing PyTorch 2.1.2 + cu121..."
pip install --upgrade pip
pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# 2. MeloTTS 클론 (이미 있으면 스킵)
if [ ! -d "MeloTTS" ]; then
  echo "[2/5] Cloning MeloTTS..."
  git clone https://github.com/myshell-ai/MeloTTS.git
else
  echo "[2/5] MeloTTS already cloned, skipping."
fi

# 3. MeloTTS 설치 (editable)
echo "[3/5] Installing MeloTTS (editable)..."
pip install -e ./MeloTTS

# 4. Korean 의존성 및 추가 패키지
echo "[4/5] Installing Korean + extra deps..."
pip install -r requirements-extra.txt

# 5. unidic (MeloTTS가 일본어용으로 요구하지만 import 시 필요)
echo "[5/5] Downloading unidic..."
python -m unidic download || echo "  (unidic download 실패 — 선택 의존성이므로 무시 가능)"

echo ""
echo "=============================================="
echo " 셋업 완료. 다음 단계: python scripts/verify_env.py"
echo "=============================================="
