#!/usr/bin/env python
"""
오디오 전처리: 리샘플링(44.1kHz), 모노화, 앞뒤 무음 trim, 피크 정규화.

사용법:
    python scripts/01_prepare_audio.py \
        --in_dir data/raw \
        --out_dir data/processed \
        --target_sr 44100
"""
import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


def process_one(in_path: Path, out_path: Path, target_sr: int,
                trim_db: float, peak_dbfs: float) -> bool:
    """한 개 파일을 처리해서 out_path에 저장. 성공 시 True."""
    try:
        # 모노 + 리샘플링 (librosa 기본은 float32)
        y, _sr = librosa.load(str(in_path), sr=target_sr, mono=True)
        if y.size == 0:
            return False

        # 앞뒤 무음 trim
        y, _ = librosa.effects.trim(y, top_db=trim_db)
        if y.size < target_sr * 0.2:  # 0.2초 미만은 스킵
            return False

        # 피크 정규화
        peak = np.max(np.abs(y))
        if peak > 0:
            target_amp = 10 ** (peak_dbfs / 20.0)  # -1dBFS → 0.891
            y = y * (target_amp / peak)

        # 클리핑 방지
        y = np.clip(y, -1.0, 1.0)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), y, target_sr, subtype="PCM_16")
        return True
    except Exception as e:
        print(f"  [skip] {in_path.name}: {e}", file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, type=Path,
                    help="원본 wav 파일들이 있는 디렉토리")
    ap.add_argument("--out_dir", required=True, type=Path,
                    help="처리된 wav가 저장될 디렉토리")
    ap.add_argument("--target_sr", type=int, default=44100,
                    help="MeloTTS 기본값 44100Hz (변경 비권장)")
    ap.add_argument("--trim_db", type=float, default=30.0,
                    help="무음으로 간주할 dB 임계값 (기본 30dB)")
    ap.add_argument("--peak_dbfs", type=float, default=-1.0,
                    help="피크 정규화 목표 dBFS (기본 -1dB)")
    ap.add_argument("--ext", default="wav",
                    help="입력 파일 확장자 (wav/flac/mp3)")
    args = ap.parse_args()

    in_files = sorted(args.in_dir.rglob(f"*.{args.ext}"))
    if not in_files:
        print(f"!! '{args.in_dir}' 아래 *.{args.ext} 파일 없음.", file=sys.stderr)
        sys.exit(1)

    print(f"입력 {len(in_files)}개 → 출력 {args.out_dir} (SR={args.target_sr}Hz)")
    n_ok = 0
    for src in tqdm(in_files, desc="processing"):
        rel = src.relative_to(args.in_dir)
        dst = args.out_dir / rel.with_suffix(".wav")
        if process_one(src, dst, args.target_sr, args.trim_db, args.peak_dbfs):
            n_ok += 1

    print(f"\n완료: {n_ok}/{len(in_files)} 파일 처리됨.")
    if n_ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
