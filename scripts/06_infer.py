#!/usr/bin/env python
"""
A/B 평가 스크립트: 같은 문장을 baseline(기본 KR 모델)과 파인튜닝 모델로 각각 합성.

사용법:
    python scripts/06_infer.py \
        --ckpt outputs/ft_kr/G_latest.pth \
        --config configs/config_4060.json \
        --sentences data/scripts/eval_sentences.txt \
        --out_dir outputs/ab_samples

결과:
    outputs/ab_samples/ft/0001.wav       ← 파인튜닝 모델
    outputs/ab_samples/baseline/0001.wav ← 기본 모델
    outputs/ab_samples/blind_mapping.csv ← 블라인드 테스트용 랜덤 라벨
"""
import argparse
import csv
import random
import shutil
import sys
from pathlib import Path


def synth_with_ckpt(ckpt_path, config_path, sentences, out_dir, device="auto",
                    language="KR", speaker_id=0, speed=1.0):
    """MeloTTS의 TTS API로 합성."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # MeloTTS는 TTS(language=..., config_path=..., ckpt_path=..., device=...)
    from melo.api import TTS
    model = TTS(language=language, config_path=str(config_path),
                ckpt_path=str(ckpt_path), device=device)
    # speaker 이름 → id — model.hps.data.spk2id 에서 얻음
    spk2id = model.hps.data.spk2id
    # 기본값: 첫 번째 스피커 사용. 사용자가 지정한 경우에는 이름으로.
    if isinstance(speaker_id, str):
        if speaker_id in spk2id:
            sid = spk2id[speaker_id]
        else:
            print(f"  [warn] speaker_id '{speaker_id}' 미존재, 첫 스피커 사용")
            sid = list(spk2id.values())[0]
    else:
        sid = list(spk2id.values())[int(speaker_id) if isinstance(speaker_id, int) else 0]

    for i, text in enumerate(sentences, 1):
        out_path = out_dir / f"{i:04d}.wav"
        model.tts_to_file(text, sid, str(out_path), speed=speed)
    return len(sentences)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path,
                    help="파인튜닝된 G 체크포인트 (.pth)")
    ap.add_argument("--config", required=True, type=Path,
                    help="학습에 쓴 config.json")
    ap.add_argument("--baseline_ckpt", type=Path,
                    default=Path("pretrained/KR/checkpoint.pth"),
                    help="비교용 기본 모델 체크포인트")
    ap.add_argument("--baseline_config", type=Path,
                    default=Path("pretrained/KR/config.json"))
    ap.add_argument("--sentences", type=Path,
                    default=Path("data/scripts/eval_sentences.txt"))
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--speaker", default="KR_FT",
                    help="파인튜닝 모델에서 쓸 speaker id / name")
    ap.add_argument("--baseline_speaker", default="KR",
                    help="기본 모델에서 쓸 speaker id / name")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--device", default="auto",
                    help="auto / cuda:0 / cpu")
    args = ap.parse_args()

    if not args.sentences.exists():
        print(f"!! 평가 문장 파일 없음: {args.sentences}", file=sys.stderr)
        sys.exit(1)
    sentences = [l.strip() for l in args.sentences.read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.strip().startswith("#")]
    if not sentences:
        print("!! 유효한 평가 문장이 없습니다.", file=sys.stderr); sys.exit(1)

    print(f"[info] {len(sentences)}개 문장을 두 모델로 합성합니다.")

    # 1. 파인튜닝 모델
    ft_dir = args.out_dir / "ft"
    print(f"[1/2] fine-tuned → {ft_dir}")
    synth_with_ckpt(args.ckpt, args.config, sentences, ft_dir,
                    device=args.device, speaker_id=args.speaker, speed=args.speed)

    # 2. 기본 모델
    if args.baseline_ckpt.exists():
        bl_dir = args.out_dir / "baseline"
        print(f"[2/2] baseline    → {bl_dir}")
        synth_with_ckpt(args.baseline_ckpt, args.baseline_config, sentences, bl_dir,
                        device=args.device, speaker_id=args.baseline_speaker,
                        speed=args.speed)
    else:
        print(f"  (baseline 체크포인트 없음: {args.baseline_ckpt} — 스킵)")

    # 3. 블라인드 테스트용 매핑 (A/B 어느 쪽이 어느 모델인지 랜덤화)
    mapping_path = args.out_dir / "blind_mapping.csv"
    rng = random.Random(42)
    with mapping_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "text", "A", "B"])
        for i, t in enumerate(sentences, 1):
            swap = rng.random() < 0.5
            a = "ft" if swap else "baseline"
            b = "baseline" if swap else "ft"
            w.writerow([f"{i:04d}", t, a, b])
    print(f"[OK] 블라인드 매핑 → {mapping_path}")
    print(f"     → {args.out_dir} 아래에서 ft/ 와 baseline/ 폴더를 비교해 들어보세요.")


if __name__ == "__main__":
    main()
