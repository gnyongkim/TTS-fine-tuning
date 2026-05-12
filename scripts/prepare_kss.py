#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KSS (Korean Single Speaker Speech) → GPT-SoVITS v2 학습 형식 변환 스크립트

다운로드 + 전처리 + train/val 분할 + .list 파일 생성을 한 번에 수행한다.

기본 동작:
  1. HuggingFace `Bingsu/KSS_Dataset` 자동 다운로드 (~3.81GB, 12,853 클립)
  2. 길이 필터 (3~10초, 변경 가능)
  3. 32kHz mono WAV로 변환 (GPT-SoVITS v2 권장 sample rate)
  4. expanded_script (숫자/약어가 한글로 풀어진 버전) 사용
  5. train.list / val.list 생성 (95:5 기본)

사용 예:
  # 기본 (HF에서 다운로드 → ~/data/kss_processed 에 출력)
  python scripts/prepare_kss.py --out-dir ~/data/kss_processed

  # 로컬에 이미 받아둔 KSS 폴더가 있을 때 (transcript.v.1.4.txt + 1/.. 4/ 폴더 구조)
  python scripts/prepare_kss.py --source local --src-dir /path/to/kss --out-dir ~/data/kss_processed

  # 길이 필터·val 비율 변경
  python scripts/prepare_kss.py --min-dur 2.0 --max-dur 12.0 --val-ratio 0.02

라이선스 주의:
  KSS는 CC BY-NC-SA 4.0 (비상업) 입니다. 본 프로젝트에서는 베이스라인 학습 검증
  용도로만 사용하며, 상용 배포 모델은 라이선스 호환 데이터셋으로 재학습합니다.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 의존성 체크
# ---------------------------------------------------------------------------
try:
    import numpy as np
    import soundfile as sf
    import librosa
    from tqdm import tqdm
except ImportError as e:
    sys.exit(
        f"[ERR] 필수 패키지 누락: {e}\n"
        "      conda activate gptsovits 후 다시 실행하세요.\n"
        "      (또는: pip install soundfile librosa tqdm numpy)"
    )

GPT_SOVITS_LANG_CODE = "ko"  # GPT-SoVITS v2 한국어 코드


# ---------------------------------------------------------------------------
# 인자 파싱
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--source", choices=["hf", "local"], default="hf",
                   help="데이터 소스. hf=HuggingFace 자동 다운로드(기본), "
                        "local=이미 받아둔 KSS 폴더 사용")
    p.add_argument("--src-dir", type=str, default=None,
                   help="(--source local 일 때만) KSS 압축 해제 폴더 경로. "
                        "transcript.v.1.4.txt 와 1/, 2/, 3/, 4/ 폴더가 들어 있어야 함")
    p.add_argument("--out-dir", type=str, default=str(Path.home() / "data" / "kss_processed"),
                   help="전처리 결과 저장 폴더 (기본: ~/data/kss_processed)")
    p.add_argument("--speaker", type=str, default="kss",
                   help="화자 이름 (.list 파일의 speaker 필드, 기본: kss)")
    p.add_argument("--sr", type=int, default=32000,
                   help="출력 샘플링 레이트 Hz (GPT-SoVITS v2 권장 32000)")
    p.add_argument("--min-dur", type=float, default=3.0,
                   help="최소 클립 길이 초 (기본 3.0)")
    p.add_argument("--max-dur", type=float, default=10.0,
                   help="최대 클립 길이 초 (기본 10.0)")
    p.add_argument("--val-ratio", type=float, default=0.05,
                   help="검증셋 비율 (기본 0.05 = 5%%)")
    p.add_argument("--seed", type=int, default=42, help="셔플 시드")
    p.add_argument("--text-field", choices=["expanded", "original", "decomposed"],
                   default="expanded",
                   help="사용할 transcript 필드 "
                        "(기본 expanded — 숫자가 한글로 풀어진 버전, TTS 학습 최적)")
    p.add_argument("--limit", type=int, default=None,
                   help="처리 클립 수 제한 (디버그용)")
    p.add_argument("--dry-run", action="store_true",
                   help="실제 변환 없이 통계만 출력")
    return p.parse_args()


# ---------------------------------------------------------------------------
# HuggingFace 소스 로더 (제너레이터)
# ---------------------------------------------------------------------------
def iter_from_hf(text_field: str):
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("[ERR] `datasets` 패키지가 필요합니다. "
                 "다음을 실행: pip install datasets")

    print("[INFO] HuggingFace에서 Bingsu/KSS_Dataset 로드 중 "
          "(최초 실행 시 3.81GB 다운로드, 캐시는 ~/.cache/huggingface/) ...")
    ds = load_dataset("Bingsu/KSS_Dataset", split="train")
    print(f"[INFO] 총 {len(ds):,} 샘플 로드 완료")

    field_map = {
        "expanded": "expanded_script",
        "original": "original_script",
        "decomposed": "decomposed_script",
    }
    text_col = field_map[text_field]

    for i, row in enumerate(ds):
        text = (row.get(text_col) or "").strip()
        if not text:
            continue
        audio = row["audio"]
        yield {
            "idx": i,
            "audio_array": np.asarray(audio["array"], dtype=np.float32),
            "sr": audio["sampling_rate"],
            "duration": float(row.get("duration") or len(audio["array"]) / audio["sampling_rate"]),
            "text": text,
        }


# ---------------------------------------------------------------------------
# 로컬 소스 로더 (Kaggle/직접 다운로드 받은 폴더)
# ---------------------------------------------------------------------------
def iter_from_local(src_dir: Path, text_field: str):
    transcript = src_dir / "transcript.v.1.4.txt"
    if not transcript.exists():
        sys.exit(f"[ERR] {transcript} 가 없습니다. "
                 "KSS 압축을 푼 폴더(1~4 폴더 + transcript.v.1.4.txt)를 가리키세요.")

    # transcript.v.1.4.txt 컬럼:
    #   0: 상대경로 (예: 1/1_0000.wav)
    #   1: original_script
    #   2: expanded_script
    #   3: decomposed_script
    #   4: english_translation
    #   5: duration (sec)
    col_idx = {"original": 1, "expanded": 2, "decomposed": 3}[text_field]

    print(f"[INFO] 로컬 KSS 폴더에서 transcript 로드: {transcript}")
    rows: list[tuple[str, str, float]] = []
    with transcript.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 6:
                continue
            rel_path, dur = parts[0].strip(), float(parts[5])
            text = parts[col_idx].strip()
            if text:
                rows.append((rel_path, text, dur))
    print(f"[INFO] 총 {len(rows):,} 라인")

    for i, (rel_path, text, dur) in enumerate(rows):
        wav_path = src_dir / rel_path
        if not wav_path.exists():
            continue
        try:
            audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        except Exception as e:
            print(f"  [WARN] 읽기 실패 {wav_path}: {e}")
            continue
        if audio.ndim == 2:
            audio = audio.mean(axis=1).astype(np.float32)  # 스테레오면 모노로
        yield {
            "idx": i,
            "audio_array": audio,
            "sr": sr,
            "duration": dur,
            "text": text,
        }


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 출력 폴더 : {out_dir}")
    print(f"[INFO] 대상 SR   : {args.sr} Hz, mono")
    print(f"[INFO] 길이 필터 : {args.min_dur}s ~ {args.max_dur}s")
    print(f"[INFO] 텍스트 필드: {args.text_field}_script")
    print(f"[INFO] 화자 이름 : {args.speaker} / 언어 코드: {GPT_SOVITS_LANG_CODE}")
    print(f"[INFO] dry-run  : {args.dry_run}")

    if args.source == "hf":
        iterator = iter_from_hf(args.text_field)
    else:
        if not args.src_dir:
            sys.exit("[ERR] --source local 일 때 --src-dir 가 필요합니다.")
        iterator = iter_from_local(Path(args.src_dir).expanduser().resolve(), args.text_field)

    items: list[tuple[str, str, str, str]] = []  # (wav_abspath, speaker, lang, text)
    n_total = n_kept = n_short = n_long = n_fail = 0

    for sample in tqdm(iterator, desc="Processing", unit="clip"):
        if args.limit and n_kept >= args.limit:
            break
        n_total += 1
        dur = sample["duration"]
        if dur < args.min_dur:
            n_short += 1
            continue
        if dur > args.max_dur:
            n_long += 1
            continue

        text = sample["text"]
        audio = sample["audio_array"]
        sr = sample["sr"]

        # 모노 변환 (혹시라도 2D가 들어오면)
        if audio.ndim == 2:
            audio = audio.mean(axis=1).astype(np.float32)

        # 리샘플
        if sr != args.sr:
            try:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=args.sr)
            except Exception as e:
                n_fail += 1
                print(f"  [WARN] 리샘플 실패 idx={sample['idx']}: {e}")
                continue
            sr = args.sr

        out_wav = wav_dir / f"kss_{sample['idx']:05d}.wav"
        if not args.dry_run:
            try:
                sf.write(out_wav, audio, sr, subtype="PCM_16")
            except Exception as e:
                n_fail += 1
                print(f"  [WARN] 저장 실패 {out_wav}: {e}")
                continue

        items.append((str(out_wav), args.speaker, GPT_SOVITS_LANG_CODE, text))
        n_kept += 1

    print()
    print(f"[STAT] 총 처리      : {n_total:,}")
    print(f"[STAT] 길이 미달    : {n_short:,} (<{args.min_dur}s)")
    print(f"[STAT] 길이 초과    : {n_long:,} (>{args.max_dur}s)")
    print(f"[STAT] 변환 실패    : {n_fail:,}")
    print(f"[STAT] 최종 저장    : {n_kept:,}")
    if n_kept == 0:
        sys.exit("[ERR] 저장된 클립이 없습니다. 필터 옵션을 확인하세요.")

    # train / val 분할
    random.seed(args.seed)
    random.shuffle(items)
    n_val = max(1, int(len(items) * args.val_ratio))
    val_items = items[:n_val]
    train_items = items[n_val:]

    if not args.dry_run:
        train_path = out_dir / "train.list"
        val_path = out_dir / "val.list"
        for path, rows in [(train_path, train_items), (val_path, val_items)]:
            with path.open("w", encoding="utf-8") as f:
                for wav, spk, lang, text in rows:
                    f.write(f"{wav}|{spk}|{lang}|{text}\n")

        # 메타 요약
        meta_path = out_dir / "dataset_info.txt"
        total_dur_min = round(sum(
            librosa.get_duration(path=Path(p).as_posix()) for p, *_ in items) / 60, 2)
        with meta_path.open("w", encoding="utf-8") as f:
            f.write(
                f"# KSS preprocessed for GPT-SoVITS v2\n"
                f"speaker        : {args.speaker}\n"
                f"language       : {GPT_SOVITS_LANG_CODE}\n"
                f"sample_rate    : {args.sr}\n"
                f"text_field     : {args.text_field}_script\n"
                f"duration_filter: {args.min_dur}s ~ {args.max_dur}s\n"
                f"total_clips    : {len(items)}\n"
                f"train_clips    : {len(train_items)}\n"
                f"val_clips      : {len(val_items)}\n"
                f"total_minutes  : {total_dur_min}\n"
                f"train_list     : {train_path}\n"
                f"val_list       : {val_path}\n"
                f"wav_dir        : {wav_dir}\n"
                f"\n# License: CC BY-NC-SA 4.0 (비상업) — 베이스라인 검증용\n"
            )
        print()
        print(f"[OK] train.list   : {train_path} ({len(train_items):,}개)")
        print(f"[OK] val.list     : {val_path}   ({len(val_items):,}개)")
        print(f"[OK] dataset_info : {meta_path}")
        print()
        print("다음 단계: GPT-SoVITS WebUI 의 1B-Fine-tune 탭에서")
        print(f"          formatted text labelled file 에 `{train_path}` 를 지정.")
    else:
        print(f"\n[DRY-RUN] 실제 파일 저장 안 함. train={len(train_items):,} val={len(val_items):,}")


if __name__ == "__main__":
    main()
