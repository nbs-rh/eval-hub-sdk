# How to run the examples

## Watching logs

**File:** [watch_job_logs.py.](watch_job_logs.py.)

The script streams new log lines to stdout and prints the final job state to stderr when the job reaches a terminal state. Exit code is 1 if the job ends in failed. Use Ctrl+C to stop early.

### 1. Configure a profile (optional, reusable)

```shell
evalhub config set base_url https://evalhub.apps.my-cluster.example.com --profile prod
evalhub config set token "$EVALHUB_TOKEN" --profile prod
evalhub config set tenant my-team --profile prod
```

### 2. Watch logs for a running job

Using a CLI profile:

When you submit a job:

```shell
uv run evalhub eval run --config eval.yaml --profile prod
# → prints something like: Job submitted: 86c904cc-cdcf-452d-b9d8-754a7d6391ec
```

```shell
cd eval-hub-sdk
uv run python examples/watch_job_logs.py "<job_id>" --profile prod
```

#### Useful flags

```shell
uv run python examples/watch_job_logs.py "<job_id>" --profile prod \
  --benchmark-index 0 \      # single benchmark only
  --tail-lines 500 \         # lines per poll
  --poll-interval 3 \        # seconds between polls
  --timestamps \             # include K8s timestamps
  --timeout 3600             # give up after 1 hour
```
