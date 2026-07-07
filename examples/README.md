# How to run the examples

## Watching logs

**File:** [watch_job_logs.py.](watch_job_logs.py.)

The script streams new log lines to stdout and prints the final job state to stderr when the job reaches a terminal state. Exit code is 1 if the job ends in failed. Use Ctrl+C to stop early.

### 1. Configure a profile (optional, reusable)

This is for the current user in namespace (tenant) `my-team` and the `default` profile:

```shell
evalhub config set base_url https://evalhub.apps.my-cluster.example.com
evalhub config set token $(oc whoami -t)
evalhub config set tenant my-team
```

### 2. Watch logs for a running job

Using a CLI profile:

When you submit a job:

```shell
uv run evalhub eval run --config eval.yaml
# → prints something like: Job submitted: 86c904cc-cdcf-452d-b9d8-754a7d6391ec
```

```shell
cd eval-hub-sdk
uv run python examples/watch_job_logs.py "<job_id>"
```

#### Useful flags

```shell
uv run python examples/watch_job_logs.py --help
```

```shell
usage: watch_job_logs.py [-h] [--profile PROFILE] [--base-url BASE_URL] [--token TOKEN] [--token-file TOKEN_FILE]
                         [--tenant TENANT] [--ca-bundle CA_BUNDLE] [--insecure] [--benchmark-index BENCHMARK_INDEX]
                         [--tail-lines TAIL_LINES] [--poll-interval POLL_INTERVAL] [--timeout TIMEOUT] [--timestamps]
                         job_id

Stream evaluation job logs until the job completes.

positional arguments:
  job_id                            Evaluation job ID to watch

options:
  -h, --help                        show this help message and exit
  --profile PROFILE                 EvalHub CLI profile name (reads ~/.config/evalhub/config.yaml)
  --base-url BASE_URL               EvalHub base URL (or set EVALHUB_BASE_URL)
  --token TOKEN                     Bearer token (or set EVALHUB_TOKEN)
  --token-file TOKEN_FILE           Read bearer token from a file (e.g. Kubernetes service account)
  --tenant TENANT                   X-Tenant header value (or set EVALHUB_TENANT)
  --ca-bundle CA_BUNDLE             Path to CA bundle for TLS verification
  --insecure                        Skip TLS certificate verification
  --benchmark-index BENCHMARK_INDEX Watch logs for a single benchmark index only
  --tail-lines TAIL_LINES           Max log lines per poll (default: 1000)
  --poll-interval POLL_INTERVAL     Seconds between polls (default: 2.0)
  --timeout TIMEOUT                 Stop watching after this many seconds
  --timestamps                      Include Kubernetes log timestamps when supported
```
