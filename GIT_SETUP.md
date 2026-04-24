# Git 초기화 & 원격 업로드 가이드

이 프로젝트를 Git으로 관리하고 GitHub(또는 다른 원격)에 올리는 전체 순서입니다.
**Mac mini의 Terminal에서 실행**하세요. (Cowork 세션 내부의 샌드박스 셸은
마운트 특성상 `.git/index.lock` 조작이 막혀서 여기서 일부 단계는 돌지 않습니다.)

---

## 0. 전제

- 프로젝트 루트: `~/path/to/melotts-fine-tuning` (사용자가 선택한 실제 경로)
- GitHub 계정: 이미 있다고 가정
- `git`, `gh`(GitHub CLI, 선택) 설치됨

---

## 1. (최초 1회) 샌드박스가 만든 `.git` 정리

Cowork가 미리 `git init`을 시도하다 권한 문제로 깨진 `.git/`이 있을 수 있습니다.
Mac Terminal에서:

```bash
cd ~/path/to/melotts-fine-tuning

# 기존 깨진 .git 제거 (경고: 커밋 히스토리가 없다는 전제)
rm -rf .git
```

> 이미 커밋이 있던 기존 레포에 덧씌우는 경우엔 이 단계를 건너뛰세요.

---

## 2. Git 사용자 정보 (최초 1회만)

```bash
git config --global user.name  "Geunyong Kim"
git config --global user.email "gnyong@mncf.io"
```

---

## 3. 초기화 & 첫 커밋

```bash
cd ~/path/to/melotts-fine-tuning

git init -b main                    # main 브랜치로 시작
git add .gitignore                  # 먼저 무시 규칙부터 반영
git add -A                          # 나머지 소스 전부 스테이징
git status                          # 올라갈 파일 목록 확인 (아래 "검증" 참고)

git commit -m "chore: bootstrap MeloTTS Korean fine-tuning project"
```

### 검증 — 절대 커밋되면 안 되는 것들

`git status` 결과에 **이 항목들이 나오면 .gitignore가 안 먹은 것입니다**:

- `MeloTTS/` (setup_env.sh가 새로 클론함)
- `data/kss/`, `data/raw/`, `data/processed/` (오디오 원본)
- `data/metadata.list`, `data/train.list`, `data/val.list`, `data/config.json`
- `pretrained/`, `outputs/` (수백 MB~GB 체크포인트)
- `*.pth`, `*.wav`, `*.onnx`, `*.bert.pt`
- `.venv/`, `__pycache__/`, `kaggle.json`

올라가야 하는 것:
- `README.md`, `GIT_SETUP.md`, `.gitignore`
- `setup_env.sh`, `requirements-extra.txt`
- `configs/config_4060.json`
- `scripts/*.py`, `scripts/*.sh`
- `data/scripts/*.txt` (녹음 스크립트 · 평가 문장 — 이건 **커밋**)

문제가 있으면 되돌리고 다시:
```bash
git rm -r --cached <잘못-추가된-경로>
git status
git commit --amend --no-edit    # 이전 커밋에 반영
```

---

## 4. GitHub 레포 생성

### 옵션 A — `gh` CLI 사용 (가장 간편)

```bash
# 최초 1회만: gh 인증
gh auth login     # GitHub.com / HTTPS / 웹 브라우저 선택

# 레포 생성 + 원격 연결 + 첫 푸시 (비공개 예시)
gh repo create melotts-fine-tuning \
    --private \
    --source=. \
    --remote=origin \
    --push \
    --description "MeloTTS Korean fine-tuning for short notification speech"
```

끝. 브라우저에서 `https://github.com/<your-id>/melotts-fine-tuning` 열어서 확인하세요.

### 옵션 B — 웹 UI로 레포 만든 뒤 수동 연결

GitHub → **New repository** → 이름 `melotts-fine-tuning` → *Create repository* → 다음을:

```bash
git remote add origin git@github.com:<your-id>/melotts-fine-tuning.git
# HTTPS를 선호하면:
# git remote add origin https://github.com/<your-id>/melotts-fine-tuning.git

git push -u origin main
```

---

## 5. 이후의 일상적 커밋 루틴

```bash
git add <변경된 파일들>
git commit -m "feat: short, what-changed message"
git push
```

커밋 메시지 prefix 권장:
- `feat:` 새 기능 (예: 새 스크립트 추가)
- `fix:` 버그 수정
- `docs:` 문서만 변경
- `chore:` 설정·의존성·기타
- `data:` 녹음 스크립트 텍스트 수정 (`data/scripts/*.txt`)

---

## 6. RTX 4060 학습 PC에서 같은 레포 쓰기

학습 머신에서 pull 해서 그대로 돌리는 흐름:

```bash
git clone git@github.com:<your-id>/melotts-fine-tuning.git
cd melotts-fine-tuning

# 환경 세팅 (여기서 MeloTTS/ 가 클론됨 — .gitignore에 있어 커밋되지 않음)
bash setup_env.sh
conda activate melotts
pip install -r requirements-extra.txt

# KSS 다운로드 & 메타데이터 생성 (한 번만)
bash scripts/00_download_kss.sh
python scripts/kss_to_metadata.py \
    --kss_dir data/kss --transcript data/kss/transcript.v.1.4.txt \
    --out data/metadata.list --speaker KR_KSS \
    --max_dur 3.0 --max_chars 30 --drop_questions

# 이후 README의 7~10단계 실행
```

---

## 7. 데이터 · 체크포인트는 어떻게 옮기나?

`.gitignore`로 제외한 것들은 다른 수단이 필요합니다:

| 종류 | 용량 | 권장 방법 |
|------|------|----------|
| KSS 데이터 | ~8GB | 학습 PC에서 `scripts/00_download_kss.sh` 직접 다운로드 (Git 불필요) |
| 사전학습 모델 | ~500MB | 학습 PC에서 `scripts/03_download_pretrained.py` 실행 |
| 파인튜닝 체크포인트 | 수 GB | Hugging Face Hub private repo 또는 Google Drive / S3 |
| ONNX 변환본 (최종) | ~200MB | GitHub Releases 또는 HF Hub |

Git LFS도 옵션이지만 HF Hub가 TTS 체크포인트 공유에는 더 표준입니다.

---

## 트러블슈팅

| 증상 | 원인 / 해결 |
|------|------------|
| `git push` 가 401/403 | HTTPS 원격이면 PAT 만들어서 비밀번호 대신 입력, 또는 `gh auth login` 후 `gh auth setup-git` |
| `! [rejected]` on first push | 원격에 이미 README가 있는 경우. `git pull --rebase origin main` 후 재시도 |
| 큰 파일 실수로 커밋됨 | `git rm --cached <file>` → `.gitignore` 보강 → `git commit --amend` (아직 push 전이면) / push 된 경우 `git-filter-repo`로 히스토리 정리 |
| macOS에서 `.DS_Store` 섞여 들어감 | 이미 `.gitignore`에 포함돼 있음. 혹시 스테이징됐다면 `git rm --cached **/.DS_Store` |

---

## 참고: 이 레포를 공개(public)로 할 때 주의

- `kaggle.json`, `.env` 등 시크릿이 커밋되지 않았는지 `git log -p | grep -iE "key|token|secret|password"` 로 한 번 훑어보세요.
- KSS 라이선스(비상업 연구용)가 걸려 있어 **데이터 자체는 재배포하지 말고** 다운로드 스크립트만 공유합니다 — 본 프로젝트의 `.gitignore`가 이미 그렇게 구성돼 있습니다.
- MeloTTS 라이선스(MIT)도 지켜서 코드 사용 시 원저자 크레딧 유지.
