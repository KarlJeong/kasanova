## 작업 방식

반드시 **Git worktree**에서 작업한다.
```bash
git worktree add .claude/worktree/{기능명} -b feature/{기능명}
cd ../kasanova-api-{기능명}
```

- `main` 브랜치에 직접 커밋하지 않는다
- 기능 단위로 worktree를 생성하고 완료 후 PR로 병합한다

---

## 프로젝트 개요

사내 직원을 위한 Slack 기반 RAG 지식 검색 시스템의 API 서버.
Slack 메시지 수신 → LangGraph 워크플로우 실행 → 응답 전달.

- **언어**: Python 3.11+
- **프레임워크**: FastAPI + LangGraph + LangChain
- **DB**: PostgreSQL (SQLAlchemy async + Alembic)
- **인터페이스**: Slack Events API

---

## 디렉토리 구조
```
app/api/              ← FastAPI 라우터
app/core/config.py    ← 환경변수 (pydantic-settings)
app/core/database.py  ← async 엔진, 세션, Base
app/models/           ← SQLAlchemy ORM 모델
app/schemas/          ← Pydantic 스키마
app/services/         ← 비즈니스 로직
app/main.py           ← 앱 생성, 라우터 등록, lifespan
```

---

## 핵심 패턴

**환경변수**: `from app.core.config import settings` — `os.environ` 직접 접근 금지

**DB 세션**: 항상 `Depends(get_db)` 패턴 — `AsyncSessionLocal` 직접 사용 금지

**서비스 클래스**: `app/services/` 아래 클래스로 분리, `__init__(self, db: AsyncSession)` 패턴 유지

**Slack 이벤트**: 반드시 즉시 `200 OK` 반환 후 `BackgroundTask`로 처리

---

## 코드 스타일

- 모든 함수에 타입 힌트 필수
- DB/외부 API 호출은 모두 async
- 예외 처리는 `HTTPException`으로 통일
- import 순서: stdlib → third-party → local
- 줄 길이 최대 100자

---

## DB 모델 추가 시

1. `app/models/`에 모델 정의 (`Base` 상속)
2. `app/models/__init__.py`에 import 추가
3. `alembic revision --autogenerate -m "설명"`
4. `alembic upgrade head`

---

## 주의사항

- Slack `bot_id` 있는 이벤트 무시 (봇 루프 방지)
- 서명 검증 실패 시 `403` 반환, 절대 우회 금지
- `Base.metadata.create_all`은 개발 전용 — 운영은 Alembic 사용
- 기획과 다르게 구현했으면 반드시 보고한다

---

## 로컬 실행
```bash
docker-compose up -d    # PostgreSQL (호스트 포트 5433)
poetry install
alembic upgrade head
uvicorn app.main:app --reload
```

---

## 테스트 케이스 수행
pytest를 통한 테스트 케이스 수행시에는 수행 필요 여부를 물을 필요없이 즉시 수행한다.