#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPT-SoVITS v2 설치 검증 스크립트
사용법:
    conda activate gptsovits
    python verify.py
"""
from __future__ import annotations
import importlib
import sys
import traceback
from pathlib import Path

PASS = "\033[1;32m[ OK ]\033[0m"
FAIL = "\033[1;31m[FAIL]\033[0m"
WARN = "\033[1;33m[WARN]\033[0m"
INFO = "\033[1;34m[INFO]\033[0m"

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    tag = PASS if ok else FAIL
    print(f"{tag} {name}{(' — ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
# 1. Python / PyTorch / CUDA
# ---------------------------------------------------------------------------
print(f"\n{INFO} ===== 1. Python / PyTorch / CUDA =====")

py_ver = sys.version.split()[0]
check("Python 버전", py_ver.startswith("3.10") or py_ver.startswith("3.11"),
      f"{py_ver} (3.10 권장)")

try:
    import torch
    check("PyTorch import", True, torch.__version__)
    check("CUDA 사용 가능", torch.cuda.is_available(),
          f"CUDA {torch.version.cuda}")
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
        check("GPU 인식", True, f"{dev} (sm_{cap[0]}{cap[1]}, {vram}GB VRAM)")
        check("RTX 4070 Ti 확인", "4070" in dev,
              dev if "4070" in dev else f"4070 아닌 GPU: {dev}")

        # 간단한 CUDA 연산 테스트
        try:
            a = torch.randn(2048, 2048, device="cuda")
            b = torch.randn(2048, 2048, device="cuda")
            c = a @ b
            torch.cuda.synchronize()
            check("CUDA matmul 테스트", True, f"output shape={tuple(c.shape)}")
        except Exception as e:
            check("CUDA matmul 테스트", False, str(e))
except ImportError as e:
    check("PyTorch import", False, str(e))

# torchaudio
try:
    import torchaudio
    check("torchaudio", True, torchaudio.__version__)
except ImportError as e:
    check("torchaudio", False, str(e))


# ---------------------------------------------------------------------------
# 2. GPT-SoVITS 핵심 의존성
# ---------------------------------------------------------------------------
print(f"\n{INFO} ===== 2. GPT-SoVITS 핵심 의존성 =====")

CORE_PKGS = [
    "transformers", "pytorch_lightning", "gradio", "librosa", "numpy",
    "scipy", "tensorboard", "funasr", "modelscope", "sentencepiece",
    "rotary_embedding_torch", "x_transformers", "ffmpeg",  # ffmpeg-python
    "fastapi",
]
for pkg in CORE_PKGS:
    try:
        m = importlib.import_module(pkg)
        ver = getattr(m, "__version__", "?")
        check(pkg, True, ver)
    except ImportError as e:
        check(pkg, False, str(e))


# ---------------------------------------------------------------------------
# 3. 한국어 처리 의존성
# ---------------------------------------------------------------------------
print(f"\n{INFO} ===== 3. 한국어 처리 의존성 =====")

KO_PKGS = {
    "g2pk2": "g2pk2",          # G2P (한국어)
    "ko_pron": "ko_pron",      # 한국어 발음 사전
    "mecab": "python_mecab_ko",  # 형태소 분석 (import 이름은 mecab)
    "jamo": "jamo",            # (선택) 자모 분해
    "opencc": "opencc",        # 중국어용이지만 의존성으로 필요
    "pyopenjtalk": "pyopenjtalk",  # 일본어용이지만 의존성으로 필요
}
for import_name, pip_name in KO_PKGS.items():
    try:
        importlib.import_module(import_name)
        check(f"{pip_name}", True, f"import={import_name}")
    except ImportError as e:
        check(f"{pip_name}", False, str(e))


# ---------------------------------------------------------------------------
# 4. 시스템 도구 (ffmpeg)
# ---------------------------------------------------------------------------
print(f"\n{INFO} ===== 4. 시스템 도구 =====")
import shutil
ffmpeg_path = shutil.which("ffmpeg")
check("ffmpeg (CLI)", ffmpeg_path is not None, ffmpeg_path or "PATH에 없음")


# ---------------------------------------------------------------------------
# 5. 사전학습 모델 / G2PW 모델 존재 확인
# ---------------------------------------------------------------------------
print(f"\n{INFO} ===== 5. 사전학습 모델 / G2PW =====")

# 자동 탐색: ~/projects/GPT-SoVITS 또는 현재 cwd 기준
candidates = [
    Path.home() / "projects" / "GPT-SoVITS",
    Path.cwd(),
    Path.cwd() / "GPT-SoVITS",
]
repo_dir = next((p for p in candidates if (p / "GPT_SoVITS" / "pretrained_models").exists()), None)

if repo_dir is None:
    check("GPT-SoVITS 저장소 탐색", False,
          "다음 경로 중 어느 것에서도 'GPT_SoVITS/pretrained_models' 폴더를 찾지 못했습니다: "
          + ", ".join(str(c) for c in candidates))
else:
    check("GPT-SoVITS 저장소 탐색", True, str(repo_dir))
    pretrained = repo_dir / "GPT_SoVITS" / "pretrained_models"
    check("pretrained_models 폴더", pretrained.exists(), str(pretrained))
    g2pw = repo_dir / "GPT_SoVITS" / "text" / "G2PWModel"
    check("G2PWModel 폴더", g2pw.exists(), str(g2pw))
    # sv (Speaker Verification) — install.sh가 sv 폴더 존재로 다운로드 완료 판단함
    sv_dir = pretrained / "sv"
    check("pretrained/sv 폴더", sv_dir.exists(), str(sv_dir))


# ---------------------------------------------------------------------------
# 결과 요약
# ---------------------------------------------------------------------------
print(f"\n{INFO} ===== 결과 요약 =====")
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = total - passed
print(f"  통과: {passed}/{total}")
if failed:
    print(f"  실패: {failed}")
    for name, ok, detail in results:
        if not ok:
            print(f"    {FAIL} {name} — {detail}")
    sys.exit(1)
else:
    print(f"  {PASS} 모든 점검을 통과했습니다. WebUI를 실행할 준비가 되었습니다.")
    print()
    print("  실행 예시:")
    print(f"    cd {repo_dir}" if repo_dir else "    cd <GPT-SoVITS 경로>")
    print("    python webui.py ko_KR")
