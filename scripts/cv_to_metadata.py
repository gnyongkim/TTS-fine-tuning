#!/usr/bin/env python
"""
Mozilla Common Voice Korean → MeloTTS metadata.list 변환 + 오디오 전처리.

Common Voice validated.tsv 주요 컬럼:
    client_id, path, sentence_id, sentence, sentence_domain,
    up_votes, down_votes, age, gender, accent, variant, locale, segment

핵심 목표:
    1. **단일 화자**(top_n 녹음 많은 client_id) 또는 사용자 지정 client_id
    2. 품질 투표 필터 (up_votes ≥ N, down_votes ≤ M)
    3. 텍스트 길이·의문문·구어체 필터
    4. mp3 → wav (44.1kHz mono, trim, normalize) 변환
    5. MeloTTS metadata.list 생성 (절대 경로, speaker=KR_CV)

사용법 (Kaggle 노트북에서):
    # Top-1 화자만, 적정 투표 통과, 최대 8초, 최대 60자 텍스트
    !python scripts/cv_to_metadata.py \\
        --tsv /kaggle/input/.../ko/validated.tsv \\
        --clips_dir /kaggle/input/.../ko/clips \\
        --out_wav_dir /kaggle/working/processed_cv \\
        --out /kaggle/working/metadata_cv.list \\
        --top_n 1 --min_up_votes 2 --max_dur 8.0 --max_chars 60

    # 특정 화자 ID 지정 + 알림 스타일 필터 (짧은 문장, 의문문 제외)
    !python scripts/cv_to_metadata.py \\
        --tsv .../validated.tsv --clips_dir .../clips \\
        --out_wav_dir /kaggle/working/processed_cv \\
        --out /kaggle/working/metadata_cv.list \\
        --speaker_id abc123... --max_chars 30 --max_dur 4.0 --drop_questions

의존성:
    pip install librosa soundfile pandas tqdm
"""
import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


# ── 텍스트 필터 규칙 (kss_to_metadata.py와 동일 기조) ────────────────────────
QUESTION_ENDERS = (
    "습니까", "나요", "까요", "어요?", "어?", "니?", "는지", "일까", "인가",
)
CASUAL_MARKERS = ("ㅋ", "ㅎ", "ㅠ", "ㅜ")


def is_question(text: str) -> bool:
    t = text.strip().rstrip("'\" ")
    if t.endswith("?"):
        return True
    return any(t.endswith(end) for end in QUESTION_ENDERS)


def looks_too_casual(text: str) -> bool:
    return any(m in text for m in CASUAL_MARKERS)


# ── TSV 파서 ────────────────────────────────────────────────────────────────
def load_tsv(path: Path):
    """Common Voice validated.tsv → dict per row."""
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield row


def pick_top_speakers(tsv_path: Path, top_n: int):
    """녹음 수 기준 상위 N 명의 client_id 반환."""
    counter = Counter()
    for row in load_tsv(tsv_path):
        counter[row["client_id"]] += 1
    print(f"\n[스캔] 전체 화자 수: {len(counter):,}")
    print(f"[스캔] 상위 {min(top_n, 10)}명 녹음 수:")
    for i, (cid, cnt) in enumerate(counter.most_common(min(top_n, 10)), 1):
        short = cid[:16] + "..." if len(cid) > 16 else cid
        print(f"  {i}. {short:20s}  {cnt:,} 클립")
    return [cid for cid, _ in counter.most_common(top_n)]


# ── 오디오 처리 ─────────────────────────────────────────────────────────────
def process_audio(
    mp3_path: Path,
    wav_path: Path,
    target_sr: int = 44100,
    trim_top_db: float = 30.0,
    peak_norm_db: float = -1.0,
) -> float:
    """mp3 → mono wav 44.1kHz, 앞뒤 무음 trim, 피크 -1dBFS 정규화.
    Returns: duration (seconds) after processing. -1 if failed.
    """
    try:
        y, sr = librosa.load(str(mp3_path), sr=target_sr, mono=True)
    except Exception as e:
        print(f"  [load fail] {mp3_path.name}: {e}", file=sys.stderr)
        return -1.0

    if len(y) == 0:
        return -1.0

    # 앞뒤 무음 trim
    y, _ = librosa.effects.trim(y, top_db=trim_top_db)
    if len(y) == 0:
        return -1.0

    # 피크 정규화
    peak = np.max(np.abs(y))
    if peak > 0:
        target_peak = 10 ** (peak_norm_db / 20)
        y = y * (target_peak / peak)

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav_path), y, target_sr, subtype="PCM_16")
    return len(y) / target_sr


# ── 메인 ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, type=Path, help="validated.tsv 경로")
    ap.add_argument("--clips_dir", required=True, type=Path,
                    help="mp3 가 들어있는 clips/ 디렉토리")
    ap.add_argument("--out_wav_dir", required=True, type=Path,
                    help="변환된 wav 저장 폴더 (없으면 생성)")
    ap.add_argument("--out", required=True, type=Path,
                    help="출력 metadata.list 경로")
    ap.add_argument("--speaker", default="KR_CV",
                    help="MeloTTS speaker 이름 (기본 KR_CV)")
    ap.add_argument("--lang", default="KR")

    # 화자 선택
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--top_n", type=int, default=1,
                       help="녹음 많은 화자 상위 N 명을 모두 포함 (기본 1, 단일 화자)")
    group.add_argument("--speaker_id", type=str,
                       help="특정 client_id 지정 (정확히 일치)")

    # 품질 투표 필터
    ap.add_argument("--min_up_votes", type=int, default=2)
    ap.add_argument("--max_down_votes", type=int, default=0)

    # 텍스트 필터
    ap.add_argument("--min_chars", type=int, default=3)
    ap.add_argument("--max_chars", type=int, default=60,
                    help="알림 스타일이면 30 권장")
    ap.add_argument("--drop_questions", action="store_true")
    ap.add_argument("--drop_casual", action="store_true")

    # 오디오 길이 필터 (변환 후 duration 기준)
    ap.add_argument("--min_dur", type=float, default=0.5)
    ap.add_argument("--max_dur", type=float, default=8.0,
                    help="알림 스타일이면 3.0~4.0 권장")
    ap.add_argument("--target_sr", type=int, default=44100)

    # 기타
    ap.add_argument("--limit", type=int, default=0,
                    help="최대 라인 수 (0 = 무제한). 디버그/소량 테스트용")
    ap.add_argument("--dry_run", action="store_true",
                    help="오디오 변환 없이 TSV 필터링만 수행해서 통계 확인")
    args = ap.parse_args()

    if not args.tsv.exists():
        sys.exit(f"!! tsv 없음: {args.tsv}")
    if not args.clips_dir.exists():
        sys.exit(f"!! clips_dir 없음: {args.clips_dir}")

    # 1) 타겟 화자 결정
    if args.speaker_id:
        targets = {args.speaker_id}
        print(f"[화자] 지정: {args.speaker_id[:16]}...")
    else:
        ids = pick_top_speakers(args.tsv, args.top_n)
        if not ids:
            sys.exit("!! TSV 비어있거나 파싱 실패")
        targets = set(ids)
        print(f"\n[화자] Top {args.top_n} 선택됨")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out_wav_dir.mkdir(parents=True, exist_ok=True)
    clips_root = args.clips_dir.resolve()
    wav_root = args.out_wav_dir.resolve()

    # 2) TSV 순회 — 먼저 후보 수집 (화자·투표·텍스트 필터 통과)
    candidates = []
    stats = Counter()
    for row in load_tsv(args.tsv):
        stats["total"] += 1

        if row["client_id"] not in targets:
            stats["drop_speaker"] += 1
            continue

        try:
            up = int(row.get("up_votes") or 0)
            down = int(row.get("down_votes") or 0)
        except ValueError:
            up, down = 0, 0
        if up < args.min_up_votes or down > args.max_down_votes:
            stats["drop_votes"] += 1
            continue

        text = (row.get("sentence") or "").strip()
        if not (args.min_chars <= len(text) <= args.max_chars):
            stats["drop_chars"] += 1
            continue

        if args.drop_questions and is_question(text):
            stats["drop_question"] += 1
            continue

        if args.drop_casual and looks_too_casual(text):
            stats["drop_casual"] += 1
            continue

        mp3 = clips_root / row["path"]
        if not mp3.exists():
            stats["drop_missing_mp3"] += 1
            continue

        # 텍스트 정규화
        text_clean = re.sub(r"\s+", " ", text).strip('"').strip("'")
        candidates.append((mp3, text_clean))

        if args.limit and len(candidates) >= args.limit:
            break

    print(f"\n[필터 결과] {len(candidates):,}개 후보 (변환 전)")

    if args.dry_run:
        print("\n[dry_run] 오디오 변환 생략. 아래 통계만 확인하세요.")
        _print_stats(stats, len(candidates))
        return

    if not candidates:
        sys.exit("!! 후보 0건. 필터를 완화하거나 --top_n 를 올리세요.")

    # 3) 오디오 변환 + metadata 작성
    print(f"\n[변환] mp3 → wav 44.1kHz mono 시작")
    with args.out.open("w", encoding="utf-8") as fout:
        for idx, (mp3, text) in enumerate(tqdm(candidates, desc="processing")):
            wav_name = f"{idx:06d}.wav"
            wav_path = wav_root / wav_name
            duration = process_audio(mp3, wav_path, target_sr=args.target_sr)

            if duration < 0:
                stats["drop_load_fail"] += 1
                continue
            if not (args.min_dur <= duration <= args.max_dur):
                stats["drop_duration"] += 1
                wav_path.unlink(missing_ok=True)
                continue

            fout.write(f"{wav_path.resolve()}|{args.speaker}|{args.lang}|{text}\n")
            stats["kept"] += 1

    _print_stats(stats, len(candidates))

    if stats["kept"] == 0:
        sys.exit("!! 남은 라인 없음. 필터를 완화하세요.")
    if stats["kept"] < 300:
        print(f"\n[!] 경고: {stats['kept']}문장은 파인튜닝에 적음. "
              f"가능하면 500문장 이상 확보 권장.", file=sys.stderr)


def _print_stats(stats: Counter, n_candidates: int):
    print("\n" + "-" * 55)
    print(f" 총 라인         : {stats['total']:,}")
    print(f" 후보 (필터 통과): {n_candidates:,}")
    print(f" 최종 저장       : {stats['kept']:,}")
    print("-" * 55)
    print(f" 제외(화자 다름)  : {stats['drop_speaker']:,}")
    print(f" 제외(투표 미달)  : {stats['drop_votes']:,}")
    print(f" 제외(글자수)     : {stats['drop_chars']:,}")
    print(f" 제외(의문문)     : {stats['drop_question']:,}")
    print(f" 제외(구어체)     : {stats['drop_casual']:,}")
    print(f" 제외(mp3 없음)   : {stats['drop_missing_mp3']:,}")
    if stats.get("drop_load_fail"):
        print(f" 제외(로드 실패)  : {stats['drop_load_fail']:,}")
    if stats.get("drop_duration"):
        print(f" 제외(길이 범위)  : {stats['drop_duration']:,}")
    print("-" * 55)


if __name__ == "__main__":
    main()
