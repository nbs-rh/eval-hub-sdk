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

```shell
cd eval-hub-sdk
uv run python examples/watch_job_logs.py eval-abc123 --profile prod
```

Using explicit flags / env vars:

```shell
export EVALHUB_BASE_URL=https://evalhub.apps.my-cluster.example.com
export EVALHUB_TOKEN="$(oc whoami -t)"   # or your API token
export EVALHUB_TENANT=my-team
uv run python examples/watch_job_logs.py eval-abc123
```

OpenShift from inside the cluster (service account token):

```shell
uv run python examples/watch_job_logs.py eval-abc123 \
  --base-url https://evalhub.evalhub.svc:8080 \
  --token-file /var/run/secrets/kubernetes.io/serviceaccount/token \
  --ca-bundle /var/run/secrets/kubernetes.io/serviceaccount/ca.crt \
  --tenant default
```

#### Useful flags

```shell
uv run python examples/watch_job_logs.py eval-abc123 --profile prod \
  --benchmark-index 0 \      # single benchmark only
  --tail-lines 500 \         # lines per poll
  --poll-interval 3 \        # seconds between polls
  --timestamps \             # include K8s timestamps
  --timeout 3600             # give up after 1 hour
```
