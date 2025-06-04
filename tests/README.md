# Tests Directory

This folder collects **all quality‑assurance assets** for the repository—unit
specs, integration checks, data fixtures, and heavyweight end‑to‑end (E2E)
scenarios. 

---

## End‑to‑End “Marketing Agent Pipeline” Test

`e2e_marketing_agent.sh` validates the entire document‑to‑landing‑page flow in a
clean container environment. 

### What the script does

1. **Build & start** the Compose stack

   * `mcp-server` — serves mocked MCP tools on port 8002.
   * `agent-service` — FastAPI application exposing the LangGraph agents on
     port 8001.
   * Volumes:

     * `./agent_service/tmp/marketing_agent_outputs` is bind‑mounted so the
       generated HTML can be accessed from the host.

2. **Wait for readiness** by polling `GET /health` on `agent-service`.

3. **POST the sample PDF** (or a user‑supplied file) together with a prompt to
   `/run_marketing_supervisor`.

4. **Validate** the JSON response and extract:

   * `last_message` — final assistant output.
   * `html_path`   — container path to the rendered marketing page.

5. **Copy** the rendered HTML from the mounted volume into
   `tests/data/<filename>.html` for later inspection or diffing.

6. **Tear down** the stack with `docker compose down -v` (even on error) so the
   local Docker environment stays clean.

> The script is written in portable Bash and depends only on
> `docker compose`, `curl`, `jq`, and `bash >= 4`.

### Running the test

```bash
# From repo root ───────────────────────────────────────────
./tests/e2e_marketing_agent.sh
```

Optional overrides:

```bash
PDF_PATH=mydoc.pdf \
TIMEOUT_SEC=90     \
QUIET=true          \
./tests/e2e_marketing_agent.sh "Write a social post for the brochure"
```

* **PDF\_PATH** — choose a different document.
* **PROMPT**   — first positional argument becomes the user prompt.
* **TIMEOUT\_SEC** — wait longer for slow machines or network pulls.
* **QUIET=true** — suppress verbose `docker compose` output.

### How Docker Compose is wired

```yaml
services:
  mcp-server:
    build: ./mcp_server
    ports: ["8002:8002"]
  agent-service:
    build: .
    depends_on: [mcp-server]
    ports: ["8001:8001"]
    volumes:
      - ./agent_service/tmp/marketing_agent_outputs:/tmp/marketing_agent_outputs
```

* `mcp-server` must be healthy before `agent-service` can download tool
  definitions at startup; `depends_on` ensures ordering.
* The shared volume gives the host a reliable path to the generated HTML file.
* No external databases or caches are required; everything is
  self‑contained.

---

## Adding More Tests

* **Unit / Integration:** place new `test_*.py` files in `tests/unit/` or
  `tests/integration/`.  They will be auto‑discovered by pytest.
* **E2E scripts:** add a sibling Bash (or Python) file, name it
  `e2e_<feature>.sh`, and document it in this README.
* **Fixtures:** drop static assets into `tests/data/` and refer to them with a
  repository‑relative path just like `sample_input.pdf`.

