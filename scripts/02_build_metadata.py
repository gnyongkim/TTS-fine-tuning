#!/usr/bin/env python
"""
녹음 WAV + 스크립트 텍스트 파일 → MeloTTS filelist (metadata.list) 생성.

규칙:
  - wav_dir 안의 파일을 이름순 정렬.
  - script 파일의 n번째 라인이 n번째 wav에 매칭.
  - 빈 라인은 스킵.
  - wav 파일 수 != 유효 라인 수 면 경고.

출력 포맷 (MeloTTS 표준, pipe-delimited):
    wav_path|speaker|language|text

사용법:
    python scripts/02_build_metadata.py \
        --wav_dir data/processed \
        --script  data/scripts/notification_scripts.txt \
        --speaker KR_FT \
        --lang    KR \
        --out     data/metadata.list
"""
import argparse
import sys
from pathlib import Path


def clean_text(t: str) -> str:
    """기본적인 텍스트 정리 — 양끝 공백, 내부 중복 공백, 따옴표 잔여 제거."""
    t = t.strip()
    # 줄바꿈/탭이 내부에 있으면 공백으로
    t = " ".join(t.split())
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", required=True, type=Path)
    ap.add_argument("--script",  required=True, type=Path)
    ap.add_argument("--speaker", default="KR_FT",
                    help="화자 ID (영숫자+_). MeloTTS config의 spk2id 키와 맞춤.")
    ap.add_argument("--lang", default="KR", choices=["KR", "EN", "ZH", "JP", "ES", "FR"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--abs_path", action=argparse.BooleanOptionalAction, default=True,
                    help="wav 경로를 절대경로로 저장 (기본: True). "
                         "MeloTTS preprocess는 다른 CWD에서 실행되므로 절대경로 권장.")
    args = ap.parse_args()

    wavs = sorted(args.wav_dir.glob("*.wav"))
    if not wavs:
        print(f"!! '{args.wav_dir}' 아래 wav 없음.", file=sys.stderr); sys.exit(1)

    lines = args.script.read_text(encoding="utf-8").splitlines()
    texts = [clean_text(l) for l in lines]
    texts = [t for t in texts if t and not t.startswith("#")]

    if len(wavs) != len(texts):
        print(f"!! wav {len(wavs)}개 vs 텍스트 {len(texts)}줄 — 수가 다릅니다.",
              file=sys.stderr)
        print(f"   짧은 쪽 기준으로 {min(len(wavs), len(texts))}개만 사용합니다.",
              file=sys.stderr)

    n = min(len(wavs), len(texts))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as f:
        for w, t in zip(wavs[:n], texts[:n]):
            wpath = str(w.resolve()) if args.abs_path else str(w)
            f.write(f"{wpath}|{args.speaker}|{args.lang}|{t}\n")

    print(f"[OK] {args.out} — {n} 라인 작성")
    print(f"     (speaker={args.speaker}, lang={args.lang})")


if __name__ == "__main__":
    main()
