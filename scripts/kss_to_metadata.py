#!/usr/bin/env python
"""
KSS (Korean Single Speaker Speech) 데이터셋을 MeloTTS metadata.list 로 변환.

KSS transcript.v.1.4.txt 각 라인 포맷 (pipe-delimited):
    <wav_rel_path>|<text_raw>|<text_normalized>|<text_decomposed>|<duration_sec>|<english>
예:
    1/1_0000.wav|그는 괜찮은 척하려고 애쓰는 것 같았다.|...|3.5|He seemed ...

옵션 B 핵심: 짧은 알림/안내 스타일 문장만 선별.
  - duration 필터:  --max_dur / --min_dur  (기본 0.5~3.0초)
  - 문자 길이 필터: --max_chars            (기본 30자)
  - 질문 제외:     --drop_questions        (기본 True; 알림은 주로 평서/명령)

사용법:
    # 전체 KSS 사용 (12,854문장, 약 12시간)
    python scripts/kss_to_metadata.py \\
        --kss_dir    /path/to/kss \\
        --transcript /path/to/kss/transcript.v.1.4.txt \\
        --out        data/metadata.list

    # 짧은 알림 스타일만 선별 (권장 — 도메인 일치 + 학습 빠름)
    python scripts/kss_to_metadata.py \\
        --kss_dir    /path/to/kss \\
        --transcript /path/to/kss/transcript.v.1.4.txt \\
        --out        data/metadata.list \\
        --max_dur 3.0 --min_dur 0.5 --max_chars 30 --drop_questions
"""
import argparse
import re
import sys
from pathlib import Path


# 알림/안내와 거리가 먼 문장을 걸러내는 간단한 규칙
QUESTION_ENDERS = ("?", "까", "죠", "니", "나요", "세요?", "가요", "니까")
EXCLAMATION = ("!",)
# 너무 구어체/감정적인 표현은 알림 도메인에서 드뭄
CASUAL_MARKERS = ("ㅋ", "ㅎ", "ㅠ", "ㅜ")


def is_question(text: str) -> bool:
    t = text.strip().rstrip("'\" ")
    if t.endswith("?"):
        return True
    # 물음표 없지만 의문형 종결어미로 끝나는 경우
    for ender in ("습니까", "나요", "까요", "어요?", "어?", "니?", "는지", "일까"):
        if t.endswith(ender):
            return True
    return False


def looks_too_casual(text: str) -> bool:
    return any(m in text for m in CASUAL_MARKERS)


def parse_transcript(path: Path):
    """Yield dicts: {rel_wav, text, duration}"""
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 5:
                print(f"  [warn] line {ln}: 필드 수 부족 ({len(parts)}), 스킵",
                      file=sys.stderr)
                continue
            rel_wav = parts[0].strip()
            text = parts[1].strip()
            try:
                duration = float(parts[4].strip())
            except ValueError:
                print(f"  [warn] line {ln}: duration 파싱 실패: {parts[4]!r}",
                      file=sys.stderr)
                continue
            yield {"rel_wav": rel_wav, "text": text, "duration": duration}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kss_dir", required=True, type=Path,
                    help="KSS 루트 (내부에 1/, 2/, 3/, 4/ 디렉토리와 wav 파일들)")
    ap.add_argument("--transcript", required=True, type=Path,
                    help="transcript.v.1.4.txt 경로")
    ap.add_argument("--out", required=True, type=Path,
                    help="출력 metadata.list 경로")
    ap.add_argument("--speaker", default="KR_KSS",
                    help="화자 ID (기본 KR_KSS)")
    ap.add_argument("--lang", default="KR")
    # 필터 옵션
    ap.add_argument("--min_dur", type=float, default=0.0,
                    help="최소 길이(초). 기본 0 (제한 없음).")
    ap.add_argument("--max_dur", type=float, default=999.0,
                    help="최대 길이(초). 알림 스타일이면 3.0 권장.")
    ap.add_argument("--max_chars", type=int, default=10_000,
                    help="텍스트 최대 글자 수. 알림 스타일이면 30 권장.")
    ap.add_argument("--drop_questions", action="store_true",
                    help="의문문 제외 (알림/안내는 평서/명령 위주이므로 권장).")
    ap.add_argument("--drop_casual", action="store_true",
                    help="ㅋㅎㅠ 등 구어체 제외.")
    ap.add_argument("--limit", type=int, default=0,
                    help="최대 라인 수 (0 = 무제한). 디버그/소량 테스트용.")
    ap.add_argument("--check_wav", action="store_true", default=True,
                    help="wav 파일 존재 여부 확인 (기본 True).")
    args = ap.parse_args()

    if not args.transcript.exists():
        print(f"!! transcript 없음: {args.transcript}", file=sys.stderr); sys.exit(1)
    if not args.kss_dir.exists():
        print(f"!! kss_dir 없음: {args.kss_dir}", file=sys.stderr); sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    kss_root = args.kss_dir.resolve()

    stats = {
        "total": 0, "kept": 0,
        "dropped_missing_wav": 0,
        "dropped_duration": 0,
        "dropped_chars": 0,
        "dropped_question": 0,
        "dropped_casual": 0,
    }

    with args.out.open("w", encoding="utf-8") as fout:
        for rec in parse_transcript(args.transcript):
            stats["total"] += 1

            # 필터 1: duration
            if not (args.min_dur <= rec["duration"] <= args.max_dur):
                stats["dropped_duration"] += 1
                continue

            # 필터 2: 글자 수
            if len(rec["text"]) > args.max_chars:
                stats["dropped_chars"] += 1
                continue

            # 필터 3: 의문문
            if args.drop_questions and is_question(rec["text"]):
                stats["dropped_question"] += 1
                continue

            # 필터 4: 구어체
            if args.drop_casual and looks_too_casual(rec["text"]):
                stats["dropped_casual"] += 1
                continue

            # 필터 5: wav 존재 확인
            wav_abs = (kss_root / rec["rel_wav"]).resolve()
            if args.check_wav and not wav_abs.exists():
                stats["dropped_missing_wav"] += 1
                continue

            # 텍스트 정규화: 양끝 따옴표 같은 잔여 제거
            text = rec["text"].strip().strip('"').strip("'")
            text = re.sub(r"\s+", " ", text)

            fout.write(f"{wav_abs}|{args.speaker}|{args.lang}|{text}\n")
            stats["kept"] += 1

            if args.limit and stats["kept"] >= args.limit:
                break

    # 통계 출력
    print("-" * 50)
    print(f" 총 라인        : {stats['total']:,}")
    print(f" 저장됨         : {stats['kept']:,}   →  {args.out}")
    print(f" 제외(duration) : {stats['dropped_duration']:,}")
    print(f" 제외(글자수)   : {stats['dropped_chars']:,}")
    print(f" 제외(의문문)   : {stats['dropped_question']:,}")
    print(f" 제외(구어체)   : {stats['dropped_casual']:,}")
    print(f" 제외(wav 없음) : {stats['dropped_missing_wav']:,}")
    print("-" * 50)

    if stats["kept"] == 0:
        print("!! 남은 라인 없음. 필터를 완화하거나 kss_dir 경로를 확인하세요.",
              file=sys.stderr)
        sys.exit(1)

    if stats["kept"] < 300:
        print(f"[!] 경고: {stats['kept']}문장은 파인튜닝에 적음. 가능하면 "
              f"필터를 완화해 500~2000문장 확보 권장.", file=sys.stderr)


if __name__ == "__main__":
    main()
