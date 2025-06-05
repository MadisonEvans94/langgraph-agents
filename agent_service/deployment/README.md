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
   │  └─ sample_input.pdf        # sample PDF to be ingested 
   ├─ e2e_marketing_agent.sh     # end to end script
   └─ README.md               
```

---

## Quick Start 

1. Ensure that you have a `.env` file at the project root with `OPENAI_API_KEY` set. There is a template file to help with this. Run `cp .env.template .env` from the project root and then fill out the key as necessary. 
2. Navigate to `agent_service/deployment`. Run `./e2e_marketing_agent.sh` to execute the marketing agent e2e pipeline. This script does the following
- calls `docker compose up --build` on the `docker-compose.yaml` file at the project root (this will spin up the agent service container and the mcp server container and set up a volume for the placement of the .html file that gets written by the agent)
- sends the `sample_input.pdf` file and user prompt to the `/run_marketing_supervisor` endpoint in the agent service. This kicks off the execution of the marketing supervisor agent
- writes the final html output of the agent to the `agent_service/deployment/data` directory 
- returns a status response to the terminal regarding the results of the run 
- tears down the containers 

> For more information on the different code and dependencies used, refer to the following READMEs 

- [agent service api](agent_service/app/README.md)
- [mock mcp server](mcp_server/README.md)
- [agent resources](agent_resources/README.md)

---

## Workflow Overview 

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

