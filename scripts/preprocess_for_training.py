#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_kss.py 결과(.list 파일) → GPT-SoVITS 학습 입력으로 변환

GPT-SoVITS 의 prepare_datasets/{1,2,3}-*.py 를 환경변수와 함께 순차 호출하여
다음 산출물을 생성한다 ($EXP_DIR 안):
    2-name2text.txt           # 음소 시퀀스
    3-bert/                   # BERT 임베딩 (.pt)
    4-cnhubert/               # HuBERT SSL 임베딩 (.pt)
    5-wav32k/                 # 32kHz 리샘플 wav
    6-name2semantic.tsv       # semantic token

사용:
    source configs/experiment.env
    python scripts/preprocess_for_training.py
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path



def env(key: str, default: str | None = None, *, required: bool = False) -> str:
    """환경변수 읽기 + 필수 변수 누락 시 에러 처리 및 종료"""
    val = os.environ.get(key, default)
    if required and not val:
        sys.exit(f"[ERR] 환경변수 {key} 가 설정되지 않았습니다. configs/experiment.env 를 source 하세요.")
    return val or ""


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱해서 환경변수대신 사용"""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", default=None, help="입력 .list 파일 (기본: $TRAIN_LIST)")
    p.add_argument("--exp-dir", default=None, help="실험 출력 폴더 (기본: $EXP_DIR)")
    p.add_argument("--exp-name", default=None, help="실험 이름 (기본: $EXP_NAME)")
    p.add_argument("--skip", choices=["text", "hubert", "semantic"], action="append", default=[],
                   help="특정 스텝 스킵 (재실행 시 유용)")
    return p.parse_args()


def run_step(label: str, script_rel: str, env_overrides: dict[str, str], gpt_sovits_dir: Path) -> None:
    """GPT-SoVITS 의 prepare_datasets/*.py 를 별도 프로세스로 실행 (환경변수 주입)."""
    script_path = gpt_sovits_dir / script_rel
    if not script_path.exists():
        sys.exit(f"[ERR] 스크립트를 찾을 수 없습니다: {script_path}")

    # GPT-SoVITS 코드는 cwd 를 GPT-SoVITS 루트로 가정 (text/, AR/, module/ import)
    # https://github.com/RVC-Boss/GPT-SoVITS/tree/main/GPT_SoVITS
    sub_env = os.environ.copy()
    sub_env.update(env_overrides)
    sub_env["PYTHONPATH"] = f"{gpt_sovits_dir}:{gpt_sovits_dir / 'GPT_SoVITS'}:" + sub_env.get("PYTHONPATH", "")
    print(f"\n{'='*78}\n[{label}] {script_path.name}\n{'='*78}")

    print("주입된 환경 변수")
    for k, v in env_overrides.items():
        print(f"  {k}={v}")
    cmd = [sys.executable, str(script_path)]
    print(f"  $ {shlex.join(cmd)}")

    proc = subprocess.run(cmd, env=sub_env, cwd=str(gpt_sovits_dir))
    if proc.returncode != 0:
        sys.exit(f"[ERR] {label} 실패 (exit code {proc.returncode})")
    print(f"[OK] {label} 완료")


def main() -> None:
    args = parse_args()

    train_list = Path(args.list or env("TRAIN_LIST", required=True))
    exp_dir = Path(args.exp_dir or env("EXP_DIR", required=True))
    exp_name = args.exp_name or env("EXP_NAME", "kss_baseline")
    gpt_sovits_dir = Path(env("GPT_SOVITS_DIR", required=True))
    wav_dir = Path(env("WAV_DIR", str(train_list.parent / "wavs")))

    if not train_list.exists():
        sys.exit(f"[ERR] {train_list} 가 없습니다. scripts/prepare_kss.py 먼저 실행하세요.")
    if not gpt_sovits_dir.exists():
        sys.exit(f"[ERR] GPT-SoVITS 폴더 없음: {gpt_sovits_dir}. setup.sh 먼저 실행하세요.")

    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] train_list   : {train_list}")
    print(f"[INFO] wav_dir      : {wav_dir}")
    print(f"[INFO] exp_dir      : {exp_dir}")
    print(f"[INFO] exp_name     : {exp_name}")
    print(f"[INFO] GPT-SoVITS   : {gpt_sovits_dir}")

    common_env = {
        "inp_text": str(train_list),
        "inp_wav_dir": str(wav_dir),
        "exp_name": exp_name,
        "opt_dir": str(exp_dir),
        "i_part": "0",
        "all_parts": "1",
        "_CUDA_VISIBLE_DEVICES": env("GPU_ID", "0"),
        "is_half": env("IS_HALF", "True"),
        "version": env("VERSION", "v2"),
    }

    # ---------- 1. BERT 텍스트 임베딩 ----------
    if "text" not in args.skip:
        run_step(
            label="1/3  Text → phoneme + BERT",
            script_rel="GPT_SoVITS/prepare_datasets/1-get-text.py",
            env_overrides={
                **common_env,
                "bert_pretrained_dir": env("PRETRAINED_BERT", required=True),
            },
            gpt_sovits_dir=gpt_sovits_dir,
        )

    # ---------- 2. HuBERT SSL 임베딩 + 32kHz wav ----------
    if "hubert" not in args.skip:
        run_step(
            label="2/3  HuBERT SSL + wav32k",
            script_rel="GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py",
            env_overrides={
                **common_env,
                "cnhubert_base_dir": env("PRETRAINED_HUBERT", required=True),
            },
            gpt_sovits_dir=gpt_sovits_dir,
        )

    # ---------- 3. semantic token ----------
    if "semantic" not in args.skip:
        run_step(
            label="3/3  Semantic token",
            script_rel="GPT_SoVITS/prepare_datasets/3-get-semantic.py",
            env_overrides={
                **common_env,
                "pretrained_s2G": env("PRETRAINED_S2G", required=True),
            },
            gpt_sovits_dir=gpt_sovits_dir,
        )

    # ---------- 산출물 점검 ----------
    print(f"\n{'='*78}\n전처리 산출물 점검\n{'='*78}")
    expected = ["2-name2text.txt", "3-bert", "4-cnhubert", "5-wav32k", "6-name2semantic.tsv"]
    for name in expected:
        p = exp_dir / name
        if p.exists():
            if p.is_dir():
                n = sum(1 for _ in p.iterdir())
                print(f"  [OK]   {p}  ({n}개 파일)")
            else:
                size_kb = p.stat().st_size / 1024
                print(f"  [OK]   {p}  ({size_kb:.1f} KB)")
        else:
            print(f"  [MISS] {p}")

    print()
    print("[DONE] 전처리 완료. 다음 단계:")
    print("  bash scripts/train_s1.sh    # GPT 학습")
    print("  bash scripts/train_s2.sh    # SoVITS 학습 (s1 학습과 별개로 진행 가능)")


if __name__ == "__main__":
    main()
