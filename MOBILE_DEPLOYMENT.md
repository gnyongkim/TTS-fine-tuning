# 모바일 배포 (ONNX) 로드맵 — 옵션 4

> 최종 목표: **파인튜닝한 MeloTTS-KR을 ONNX로 변환해 모바일 앱에 on-device 탑재**,
> 합성 텍스트는 **완전히 동적**(사용자·서버 입력)이라는 전제.

이 문서는 학습 이후의 작업만 다룹니다. 학습 자체는 `README.md` 참고.

---

## 1. 현실 점검

### ✅ 확인된 사실
- **MeCab은 한국어에 쓰이지 않음**. MeloTTS의 `melo/text/korean.py`는 `g2pkk` + `jamo` + HF `transformers`(BERT) 세 가지만 사용. 초기에 우려했던 MeCab 의존성은 일본어 경로 전용.
- MeloTTS 모델 본체(VITS/generator)는 ONNX export 자체는 가능. 커뮤니티에 성공 사례 있음.
- 한국어 지원 ONNX 포크 존재: `201831771214/MeloTTS-ONNX` (참고용).

### ⚠️ 블로커
1. **g2pkk** — 순수 Python 규칙 + 사전 기반. C/Java/Kotlin/Swift 포트 없음. **직접 포팅이 필요**.
2. **`kykim/bert-kor-base`** — BERT-base (~110MB). 한국어 BERT 임베딩이 MeloTTS 입력으로 들어감. 이것도 ONNX로 변환해 앱에 함께 싣거나 서버에 두고 API 호출해야 함.
3. **공식 export 도구 부재** — MeloTTS 리포의 issue #98 ONNX 관련 스레드가 미해결. 스스로 export 스크립트를 써야 함.

---

## 2. 전체 파이프라인 (런타임)

앱에서 한 문장 합성 시 실행 순서:

```
입력 텍스트 (동적)
    │
    ▼
[1] 텍스트 정규화       ← 순수 문자열 처리 (Swift/Kotlin 직접)
    │                    숫자·영어·특수문자 한글화 등
    ▼
[2] G2P 변환            ← **g2pkk 포팅본** (핵심 작업)
    │                    한글 → 자모/음소 시퀀스
    ▼
[3] BERT 임베딩         ← kykim/bert-kor-base (ONNX)
    │                    문맥 벡터 생성
    ▼
[4] VITS Generator       ← MeloTTS 본체 (ONNX)
    │                    음소 + BERT → 멜로 → 파형
    ▼
[5] PCM → 오디오 재생    ← 플랫폼 오디오 API
```

[3]과 [4]가 ONNX Runtime Mobile / CoreML / NNAPI로 돌고,
[1][2][5]는 네이티브 코드.

---

## 3. 단계별 작업 목록

### Phase A — 학습 완료 후 ONNX export
| 단계 | 산출물 | 참고 |
|------|--------|------|
| A1 | VITS Generator → ONNX (`generator.onnx`) | `201831771214/MeloTTS-ONNX` 포크의 export 스크립트 참고 |
| A2 | BERT → ONNX (`bert_kor.onnx`) | `transformers`의 `onnx.export` 표준 절차 |
| A3 | 두 ONNX 모두 `onnxruntime` CPU로 문장 1개 합성 검증 | 품질이 PyTorch 버전과 거의 같아야 함 |
| A4 | INT8 동적 양자화 검증 | VITS는 음질 저하 주의 — 옵션으로만 |

### Phase B — g2pkk 포팅 (프로덕트 팀 작업)
| 단계 | 산출물 | 메모 |
|------|--------|------|
| B1 | g2pkk 규칙 추출 | `g2pkk`는 Python 규칙 함수들 + CSV 사전. 규칙은 선형이라 이식 가능 |
| B2 | Kotlin/Swift 포트 | **이 단계가 전체 일정의 가장 큰 리스크** |
| B3 | PyTorch g2pkk vs 포팅본 출력 diff 테스트 | 1000문장 A/B로 음소 시퀀스 일치율 ≥ 99% 목표 |
| B4 | jamo 변환은 자체 구현 (유니코드 0xAC00 범위 산술) | 10~50줄짜리라 간단 |

### Phase C — 런타임 통합
| 단계 | 산출물 |
|------|--------|
| C1 | iOS: CoreML 변환 또는 ONNX Runtime Mobile iOS 패키지 |
| C2 | Android: NNAPI/GPU 지원 ORT Mobile AAR |
| C3 | 통합 API: `synthesize(text: String) -> Float[]` |
| C4 | 모델 다운로드 전략 (앱 번들 vs 런타임 다운로드). BERT 110MB + VITS ~50MB = 번들 ~180MB |

---

## 4. 리스크와 대안

| 리스크 | 대안 |
|-------|------|
| g2pkk 포팅 공수가 너무 큼 | **하이브리드**: G2P만 서버 API로, VITS는 on-device. 초기 릴리즈에 권장 |
| BERT 110MB가 번들에 부담 | 서버에서 임베딩만 반환하는 경량 API로 분리 (지연 비용 발생) |
| VITS ONNX 품질 저하 | FP32 유지, INT8 양자화는 금지 / OR CoreML fp16만 사용 |
| 앱 콜드스타트 지연 | 첫 기동 시 모델 preload 스레드, 또는 lazy load |

---

## 5. "먼저 증명부터" 체크리스트

학습에 시간 쓰기 전, **1일짜리 PoC**로 다음을 확인하세요:

1. `myshell-ai/MeloTTS-Korean` 원본 체크포인트로 ONNX export 성공
2. `onnxruntime` (CPU) 로 한국어 문장 합성 → 파형 저장
3. 합성 품질이 PyTorch 원본과 **거의** 같은지 귀로 확인
4. Android/iOS 에뮬레이터에서 ORT Mobile로 동일 ONNX 구동

4가지가 다 되면 파인튜닝에 투자할 가치가 확실합니다. 하나라도 막히면 대안(예: Piper TTS — 경량화·모바일 친화적이지만 품질 다름)을 재고.

---

## 6. 커뮤니티 참고 링크 (학습 이후 참조)

- MeloTTS 공식: https://github.com/myshell-ai/MeloTTS
- ONNX 포크 (한국어 포함): https://github.com/201831771214/MeloTTS-ONNX
- ONNX 포크 (중영만): https://github.com/season-studio/MeloTTS-ONNX
- ONNX Runtime Mobile 문서: https://onnxruntime.ai/docs/tutorials/mobile/
- `kykim/bert-kor-base`: https://huggingface.co/kykim/bert-kor-base
- g2pkk: https://github.com/Kyubyong/g2pK (origin) / https://pypi.org/project/g2pkk/

> ※ 위 커뮤니티 포크들은 원 소유자 업데이트·라이선스를 반드시 확인 후 활용하세요.

---

## 7. 다음 액션

이 프로젝트의 `scripts/` 아래에는 아직 ONNX export 스크립트가 없습니다.
**파인튜닝 완료(README 9단계) 후**에 다음 스크립트를 추가 예정:

- `scripts/07_export_onnx.py` — VITS generator export
- `scripts/08_export_bert_onnx.py` — `kykim/bert-kor-base` export
- `scripts/09_test_onnx_inference.py` — CPU ONNX Runtime으로 파이프라인 검증

지금은 **파인튜닝 파이프라인을 먼저 검증**(옵션 B / KSS)하고, Phase A PoC로 넘어가는 게 투자 대비 리스크가 가장 낮습니다.
