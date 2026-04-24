#!/usr/bin/env python
"""
환경 검증 스크립트.
GPU · VRAM · PyTorch CUDA · MeloTTS import · Korean G2P 가 모두 동작하는지 확인.

사용법:
    python scripts/verify_env.py
"""
import sys
import subprocess


def check(label, fn):
    print(f"[ .. ] {label}", end="", flush=True)
    try:
        msg = fn()
        print(f"\r[ OK ] {label} — {msg}")
        return True
    except Exception as e:
        print(f"\r[FAIL] {label}")
        print(f"         └─ {type(e).__name__}: {e}")
        return False


def check_python():
    v = sys.version_info
    assert v >= (3, 9), f"Python {v.major}.{v.minor} — 3.9+ 권장"
    return f"Python {v.major}.{v.minor}.{v.micro}"


def check_torch_cuda():
    import torch
    assert torch.cuda.is_available(), "CUDA unavailable"
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    ver = torch.version.cuda
    return f"torch={torch.__version__}, cuda={ver}, gpu={name} ({total:.1f}GB)"


def check_vram_warning():
    import torch
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total < 7.5:
        return f"VRAM {total:.1f}GB — 매우 타이트. batch_size=1, segment_size=4096 필수"
    if total < 10:
        return f"VRAM {total:.1f}GB — batch_size=2, segment_size=8192 권장 (config_4060.json)"
    return f"VRAM {total:.1f}GB — 충분함. 기본 config 사용 가능"


def check_melo_import():
    import melo  # noqa: F401
    from melo.api import TTS  # noqa: F401
    return "melo + melo.api.TTS import 성공"


def check_korean_g2p():
    from melo.text import korean
    # 실제 한국어 G2P 시도
    text = "안녕하세요. 주문이 완료되었습니다."
    try:
        norm, phones, tones, word2ph = korean.g2p(text)
        assert len(phones) > 0
        return f"g2p OK ({len(phones)} phones) — 예: {phones[:8]}"
    except Exception as e:
        raise RuntimeError(f"korean.g2p 호출 실패: {e}")


def check_librosa_soundfile():
    import librosa, soundfile  # noqa: F401
    return f"librosa={librosa.__version__}, soundfile={soundfile.__version__}"


def check_huggingface_hub():
    from huggingface_hub import HfApi  # noqa: F401
    return "huggingface_hub OK"


def main():
    print("=" * 60)
    print(" MeloTTS Fine-tuning — Environment Verification")
    print("=" * 60)

    results = []
    results.append(check("Python version",          check_python))
    results.append(check("PyTorch + CUDA",          check_torch_cuda))
    results.append(check("VRAM advisory",           check_vram_warning))
    results.append(check("MeloTTS import",          check_melo_import))
    results.append(check("Korean G2P (g2pkk)",      check_korean_g2p))
    results.append(check("librosa / soundfile",     check_librosa_soundfile))
    results.append(check("huggingface_hub",         check_huggingface_hub))

    print("-" * 60)
    ok = sum(results)
    total = len(results)
    print(f" 결과: {ok}/{total} 통과")
    if ok == total:
        print(" ✅ 모든 검사 통과 — 다음 단계(03_download_pretrained.py)로 진행.")
    else:
        print(" ❌ 일부 검사 실패 — 위 에러 메시지를 참고해 고친 뒤 다시 실행.")
        sys.exit(1)


if __name__ == "__main__":
    main()
