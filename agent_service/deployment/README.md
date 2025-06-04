# Deployment 

`agent_resources/deployment/` holds everything required to spin up the full
**marketing‑agent pipeline** in Docker, feed it an example PDF, and retrieve the
rendered landing‑page HTML.  Locating these artefacts under *deployment* keeps
runtime code clean and avoids implying that the Bash driver is part of the unit
/ integration test harness.

```text
agent_resources/
└─ deployment/
   ├─ data/
   │  └─ sample_input.pdf        # sample PDF used by the smoke test
   ├─ e2e_marketing_agent.sh     # driver script
   └─ README.md                  # this document
```

---

## Marketing‑Agent Pipeline

`e2e_marketing_agent.sh` exercises the end‑to‑end flow **“PDF → supervisor →
sub‑agent → HTML”** inside a disposable Compose stack.

### Workflow overview

1. **Build and start Compose stack**

   | Service         | Port | Role                                                  |
   | --------------- | ---- | ----------------------------------------------------- |
   | `mcp-server`    | 8002 | Serves mocked MCP tools over Server‑Sent Events       |
   | `agent-service` | 8001 | FastAPI app that hosts the LangGraph marketing agents |

   A bind mount `./agent_service/tmp/marketing_agent_outputs` surfaces the
   generated HTML file to the host.

2. **Health probe** – the script polls `GET /health` on `agent-service` until a
   200 OK is returned or the timeout (`TIMEOUT_SEC`, default 40 s) expires.

3. **Submit job** – a multipart `POST` uploads the target PDF (`PDF_PATH`) and
   prompt to `/run_marketing_supervisor`.

4. **Validate response** – the reply is parsed as JSON.  The script prints
   `last_message` (the supervisor’s final message) and reads `html_path`, the
   container location of the rendered page.

5. **Copy artefact** – if `html_path` exists, the script copies the file from
   the bind mount into `agent_resources/deployment/data/` so that input and
   output live together.

6. **Clean up** – a `trap` always runs `docker compose down -v`, removing
   containers, networks, and volumes even on error or Ctrl‑C.

### Running the script

```bash
# From repository root
./agent_resources/deployment/e2e_marketing_agent.sh
```

Optional overrides:

```bash
PDF_PATH=docs/brochure.pdf \
TIMEOUT_SEC=90            \
QUIET=true                \
./agent_resources/deployment/e2e_marketing_agent.sh "Generate social posts"
```

* `PDF_PATH` – path to a different PDF document.
* First positional argument – prompt sent to the supervisor.
* `TIMEOUT_SEC` – seconds to wait for stack health.
* `QUIET=true` – suppresses verbose Compose output.

### Compose excerpt (reference only)

```yaml
services:
  mcp-server:
    build:
      context: ./mcp_server
      dockerfile: Dockerfile.mcp-server
    ports: ["8002:8002"]
    env_file: [.env]

  agent-service:
    build:
      context: .
      dockerfile: Dockerfile.agent-service
    depends_on: [mcp-server]
    ports: ["8001:8001"]
    env_file: [.env]
    volumes:
      - ./agent_service/tmp/marketing_agent_outputs:/tmp/marketing_agent_outputs
```

No external databases or caches are required; the stack is entirely
self‑contained.

---

## Extending the smoke‑test suite

* Place additional end‑to‑end drivers beside `e2e_marketing_agent.sh` and
  document them here.
* Store large or binary fixtures under `agent_resources/deployment/data/`.
* **Unit / integration tests** that follow pytest conventions should remain in
  the top‑level `tests/` folder so CI discovery continues to work as before.
