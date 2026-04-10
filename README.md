# rleaas — Release SDK

Python SDK for the **Release (RLEaaS)** platform by Centific — RL Environments as a Service.

## Installation

```bash
pip install rleaas
```

## API Key

| Setup | API key needed? | How to get it |
|---|---|---|
| **Local dev server** (`http://localhost:8000`) | No | Run the AgentWork-Simulator locally — no auth required |
| **Hosted / production** | Yes | Log in to the Release dashboard → **Settings → API Keys** → create a key |

Set the key as an environment variable so you never hardcode it:

```bash
export RLEAAS_API_KEY="rleaas_sk_your_key_here"
```

The SDK reads it automatically:

```python
import rleaas

client = rleaas.Client()                          # reads RLEAAS_API_KEY from env
# or
client = rleaas.Client(api_key="rleaas_sk_...")   # pass explicitly
```

### Local development (no key required)

```python
import rleaas

client = rleaas.Client(base_url="http://localhost:8000")
print(client.ping())
# {'message': 'RL Environment & Agent API', 'version': '1.0.0', ...}
```

## Sub-clients

| Attribute | Purpose |
|---|---|
| `client.Environment` | Create and manage simulation environments |
| `client.Tools` | Register and configure agent tools |
| `client.Agent` | Register and export trained agents |
| `client.Verifier` | Define scoring verifiers (rule-based, LLM judge, composite) |
| `client.Scenario` | Create and browse training scenarios |
| `client.ScenarioSuite` | Organize scenarios into training/evaluation suites |
| `client.TrainingJob` | Launch and monitor GRPO/PPO/DQN/A2C training runs |
| `client.Evaluation` | Run evaluations and retrieve rollouts |
| `client.Metrics` | Query KPIs and training metrics |
| `client.AuditLog` | Access audit logs and governance configuration |

## Example

```python
import rleaas

client = rleaas.Client()   # reads RLEAAS_API_KEY from environment

# Create environment
env = client.Environment.create(name="FinSim-Prod-v1", vertical="FinSim")
env.wait_until_ready()

# Create verifier
rule_v = client.Verifier.create(
    name="AML Compliance Check",
    verifier_type="rule_based",
    environment="FinSim-Prod-v1",
    config={
        "conditions": ["'run_aml_check' in trajectory.tool_calls"],
        "condition_logic": "AND",
        "reward_on_pass": 1.0,
        "reward_on_fail": 0.0,
    },
)

# Train
job = client.TrainingJob.run(
    environment_name="FinSim-Prod-v1",
    algorithm="GRPO",
    config={"episodes": 10000, "max_steps_per_episode": 20},
    verifier_ids=[rule_v.id],
)
job.wait_until_complete()
best = job.get_best_checkpoint()

# Evaluate
eval_job = client.Evaluation.run(
    agent_checkpoint_id=best["id"],
    scenario_suite_id="suite_eval_01",
    verifier_ids=[rule_v.id],
)
report = eval_job.wait_until_complete()
print(report["overall_score"])
```

## Examples

Clone or download the `examples/` folder and run them in order:

| File | What it shows |
|---|---|
| `examples/quickstart.py` | Connect, ping, list environments and tools |
| `examples/create_environment.py` | Create env → verifier → scenarios → suites |
| `examples/verifiers.py` | All 4 verifier types (rule, trajectory, LLM, composite) |
| `examples/train_agent.py` | Launch GRPO training, monitor progress, get best checkpoint |
| `examples/evaluate_agent.py` | Run evaluation, compare rollouts, export audit report |

```bash
# Install the SDK
pip install rleaas

# Run examples against a local server
python examples/quickstart.py
python examples/create_environment.py
python examples/verifiers.py
python examples/train_agent.py
python examples/evaluate_agent.py
```

## Async support

```python
async with rleaas.AsyncClient() as client:
    status = await client.ping()
```

## License

MIT
