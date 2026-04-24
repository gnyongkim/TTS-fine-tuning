#!/usr/bin/env python
"""
HuggingFace Hub에서 MeloTTS-Korean 사전학습 체크포인트 다운로드.

필요한 파일:
  - checkpoint.pth  (Generator)  ← --pretrain_G 로 사용
  - config.json     (사전학습 config — 참고용)

※ MeloTTS는 Discriminator/DUR 체크포인트를 공개하지 않습니다.
   파인튜닝 시 D와 DUR은 랜덤 초기화로 출발하며, G만 warm-start 됩니다.
   이 경우 초기 몇 epoch 동안 discriminator가 adversarial loss에 쓸만해질 때까지
   generator quality가 잠깐 나빠졌다가 회복되는 패턴이 정상입니다.

사용법:
    python scripts/03_download_pretrained.py --out pretrained/KR
"""
import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO = "myshell-ai/MeloTTS-Korean"
FILES = ["checkpoint.pth", "config.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pretrained/KR", type=Path)
    ap.add_argument("--repo", default=REPO)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for fname in FILES:
        print(f"[download] {args.repo}/{fname}")
        try:
            local = hf_hub_download(
                repo_id=args.repo,
                filename=fname,
                local_dir=str(args.out),
                local_dir_use_symlinks=False,
            )
            print(f"  → {local}")
        except Exception as e:
            print(f"!! '{fname}' 다운로드 실패: {e}", file=sys.stderr)
            print(f"   HuggingFace 리포에 해당 파일이 없을 수 있습니다.",
                  file=sys.stderr)
            print(f"   브라우저에서 확인: https://huggingface.co/{args.repo}/tree/main",
                  file=sys.stderr)

    print(f"\n[OK] 사전학습 파일 저장: {args.out}")
    print(f"     --pretrain_G={args.out / 'checkpoint.pth'} 로 학습 시 사용.")


if __name__ == "__main__":
    main()
