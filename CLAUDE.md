## 프로젝트 개요

KasaNova는 **사내 직원을 위한 Slack 기반 RAG 지식 검색 시스템**입니다.
사용자가 Slack에서 질의하면 사내 문서를 검색하여 답변을 생성합니다.

- **언어**: Python 3.11+
- **프레임워크**: FastAPI
- **AI**: LangGraph + LangChain + Ollama (LLM 서빙)
- **DB**: PostgreSQL (SQLAlchemy async + Alembic)
- **인터페이스**: Slack Events API

---

## 디렉토리 구조 및 역할
```
app/api/slack.py          ← Slack 이벤트 수신 라우터 (서명 검증, 라우팅)
app/core/config.py        ← 환경변수 (pydantic-settings). settings 싱글톤 사용
app/core/database.py      ← async 엔진, 세션, Base 정의
app/models/               ← SQLAlchemy ORM 모델 (테이블 정의)
app/schemas/              ← Pydantic 스키마 (요청/응답 검증)
app/services/             ← 비즈니스 로직 (DB 조작, Slack API 호출)
app/main.py               ← FastAPI 앱 생성, 라우터 등록, lifespan
```

---

## 핵심 패턴 및 규칙

### 1. Slack 이벤트 처리 — 반드시 즉시 200 반환

Slack은 3초 내 응답이 없으면 재시도합니다. 반드시 아래 패턴을 따르세요.
```python
@router.post("/slack/events")
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    # 처리 로직은 반드시 BackgroundTask로
    background_tasks.add_task(handle_message_event, event, db)
    return {"ok": True}  # 즉시 반환
```

### 2. DB 세션 — Depends(get_db) 패턴
```python
async def some_endpoint(db: AsyncSession = Depends(get_db)):
    ...
```

`AsyncSessionLocal`을 직접 사용하지 마세요. 항상 `get_db` 의존성을 통해 세션을 주입하세요.

### 3. 서비스 클래스 패턴

비즈니스 로직은 `app/services/` 아래 클래스로 분리합니다.
```python
class ConversationService:
    def __init__(self, db: AsyncSession):
        self.db = db
```

### 4. 환경변수 접근
```python
from app.core.config import settings

settings.slack_bot_token  # 이렇게 접근
```

`os.environ`을 직접 사용하지 마세요.

### 5. RAG 검색

RAG 검색은 OpenSearch를 통해 수행합니다. OpenSearch는 BM25(inverted index) + k-NN(vector) + RRF를 내장으로 제공하므로 별도 라이브러리 불필요.

- LangGraph `retrieve` 노드에서 `OPENSEARCH_URL`로 직접 호출
- 문서 인덱싱은 별도 스크립트(`scripts/ingest.py`)로 실행 (서버 코드와 무관)
- 커넥션 설정: `settings.opensearch_url`, `settings.opensearch_index` 사용

### 6. LangGraph 연결 위치

`app/api/slack.py`의 `handle_message_event` 내 `# TODO: LangGraph 호출` 주석 위치를 대체합니다.

**현재 (placeholder)**
```python
# TODO: LangGraph 호출
answer = f"[KasaNova] '{text}' 에 대한 답변입니다. (LangGraph 연결 전)"
```

**Phase 2 연결 시 (예시)**
```python
from app.graph.workflow import kasanova_graph

result = await kasanova_graph.ainvoke({
    "query": text,
    "conversation_history": history,
})
answer = result["answer"]
sources = result["sources"]
```

LangGraph 워크플로우는 `app/graph/` 디렉토리 아래에 구현하며, RAG 논의 시 함께 작업합니다.

---

## 코드 스타일

- **타입 힌트 필수**: 모든 함수 시그니처에 타입 힌트 작성
- **async/await**: DB 및 외부 API 호출은 모두 비동기로 작성
- **예외 처리**: `HTTPException`으로 통일 (FastAPI 표준)
- **import 순서**: stdlib → third-party → local (isort 기준)
- **줄 길이**: 최대 100자

---

## 자주 하는 작업

### 새 API 엔드포인트 추가
1. `app/api/` 아래 라우터 파일 생성 또는 기존 파일에 추가
2. `app/main.py`에 `app.include_router(...)` 등록
3. 필요한 스키마는 `app/schemas/`에 Pydantic 모델로 정의

### DB 모델 추가
1. `app/models/`에 SQLAlchemy 모델 정의 (`Base` 상속)
2. `alembic revision --autogenerate -m "설명"`으로 마이그레이션 생성
3. `alembic upgrade head` 적용

### 새 서비스 로직 추가
1. `app/services/`에 클래스 생성
2. `__init__(self, db: AsyncSession)` 패턴 유지
3. 엔드포인트에서 `Depends(get_db)`로 세션 주입 후 서비스 인스턴스화

---

## 주의사항

- Slack `bot_id`가 있는 이벤트는 무시 (봇 루프 방지)
- Slack `subtype`이 있는 메시지 이벤트는 무시 (메시지 편집/삭제 등)
- 서명 검증 실패 시 `403` 반환 (절대 우회 금지)
- `Base.metadata.create_all`은 개발 환경 전용 — 운영은 반드시 Alembic 사용

---

## 로컬 실행
```bash
docker-compose up -d    # PostgreSQL
poetry install
alembic upgrade head
uvicorn app.main:app --reload
```