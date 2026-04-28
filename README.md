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

### Training + Simulation configuration

Use this simple flow:

1. Put training settings in your `config.json` under `training` as an array of entries.
2. Run `python -m rleaas.training_cli` commands to start and manage jobs.

Quick helper option:

```bash
python -m rleaas.training_cli start
# status / rerun / operations:
python -m rleaas.training_cli status --job-id <job_id>
Eg: 
python -m rleaas.training_cli status --job-id train_8f3a2b1c
python -m rleaas.training_cli list
python -m rleaas.training_cli list --ids-only
python -m rleaas.training_cli wait --job-id <job_id>
python -m rleaas.training_cli metrics --job-id <job_id>
python -m rleaas.training_cli checkpoints --job-id <job_id>
python -m rleaas.training_cli rollouts --job-id <job_id>
python -m rleaas.training_cli cancel --job-id <job_id>

# if config has multiple training entries:
python -m rleaas.training_cli start --training 3
python -m rleaas.training_cli start --training 3,4,5
python -m rleaas.training_cli start --all
```
### `config.json` training schema

Place training config under top-level `training` key in `config.json` using this format only:
- `"training": [ { ... }, { ... } ]`
- each entry must include a `training` identifier (for example: `1`, `2`, `3`)

- `training` (required, integer/string): run identifier (`1`, `2`, `3`, etc.).
- `environment_name` (required, string): target environment name to train.
- `name` (optional, string): run name shown in training history/UI.
- `description` (optional, string): objective/notes for the run.
- `agent_id` (optional, string): agent/model id to use for training.
- `scenario_id` (optional, string): scenario/suite id for the run.
- `verifier_ids` (optional, array of strings): verifier ids to evaluate rewards/scoring.
- `algorithm` (optional, string, default `PPO`): one of `GRPO`, `PPO`, `SAC`, `DQN`, `A2C`, `A3C`, `TD3`, `DDPG`, `SLM`.
- `max_steps` (optional, integer, default `200` in README flow): max steps per episode.
- `episodes` (optional, integer, default `100` in README flow): number of episodes.
- `reward_fn` (optional): reward function reference in one of these formats:
  - string: `"rewards/finsim_reward.py"` or inline expression/function string
  - object path form: `{ "path": "rewards/finsim_reward.py" }`
  - object inline form: `{ "inline": "reward = ..." }`
- `simulation` (optional, object):
  - `speed` (optional, positive number)
  - `seed` (optional, non-negative integer)
  - `episode_settings` (optional, object):
    - `num_episodes` (optional, positive integer)
    - `max_steps` (optional, positive integer)
- `config` (optional, object): additional backend config, for example `learning_rate`, `batch_size`, `checkpoint_interval`.

Example:

```json
{
  "training": [
    
    {
      "training": 1,
      "name": "finsim-ppo-run-007",
      "description": "PPO training run for FinSim baseline.",
      "environment_name": "Demo-FinSim-Env-aks-Demo",
      "agent_id": "",
      "scenario_id": "",
      "verifier_ids": [],
      "algorithm": "PPO",
      "max_steps": 200,
      "episodes": 500,
      "reward_fn": {
        "path": "rewards/finsim_reward.py"
      },
      "simulation": {
        "speed": 1.5,
        "seed": 42,
        "episode_settings": {
          "num_episodes": 500,
          "max_steps": 200
        }
      },
      "config": {
        "learning_rate": 0.0003,
        "batch_size": 128,
        "checkpoint_interval": 100
      }
    },
    {
      "training": 2,
      "name": "finsim-ppo-run-008",
      "description": "PPO training run for FinSim baseline.",
      "environment_name": "Demo-FinSim-Env-aks-Demo",
      "agent_id": "",
      "scenario_id": "",
      "verifier_ids": [],
      "algorithm": "PPO",
      "max_steps": 200,
      "episodes": 500,
      "reward_fn": {
        "path": "rewards/finsim_reward.py"
      },
      "simulation": {
        "speed": 1.5,
        "seed": 43,
        "episode_settings": {
          "num_episodes": 500,
          "max_steps": 200
        }
      },
      "config": {
        "learning_rate": 0.0003,
        "batch_size": 128,
        "checkpoint_interval": 100
      }
    }
  ]
}
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
