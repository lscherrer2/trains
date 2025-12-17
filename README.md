# trains

Lightweight building blocks for describing and stepping through 
**densely-connected train track systems**.

This project started because there aren't any train frameworks or simulation libraries out there,
(especially for **tightly interconnected layouts**). I originally wrote this to simulate a model 
train system in the real world.

A **system** is set up as follows:
- Model the system as **switches**, and each switch has three **branches** (`approach`, `through`, `diverging`)
- Branches are connected together with **tracks**
- Place **trains** on the network and advance them with `step(dt)`

---

## Setup

### Requirements
- Python **3.13+** (see `pyproject.toml`)
- Dependencies include `torch` and `torch-geometric` (these can be platform-specific; install your preferred PyTorch build as needed)

### Install (recommended)
This repo uses `uv` for reproducible installs.

```bash
./script/install.sh
```

### Run tests
```bash
./script/test.sh
```

---

## Quickstart

Create a system from JSON, then step the simulation.

```python
import json
from trains.env import System

with open("test/data/system.json") as f:
    system_json = json.load(f)

system = System.from_json(system_json)

# advance all trains by dt seconds
system.step(dt=0.5)

# or step an individual train
system.trains[0].step(dt=1.0)
```

Switches can change state during simulation (e.g., route `approach -> diverging` instead of `approach -> through`).

```python
switch_b = system.switch_map["B"]
switch_b.state = True
system.step(dt=1.0)
```

---

## JSON format (example)

The most convenient way to describe a layout is via a JSON spec.
Here’s a minimal example (the same structure as `test/data/system.json`):

```json
{
	"switches": ["A", "B"],
	"tracks": [
		{
			"from_": { "switch": "A", "type_": "through" },
			"to": { "switch": "B", "type_": "approach" },
			"length": 10.0
		},
		{
			"from_": { "switch": "A", "type_": "diverging" },
			"to": { "switch": "B", "type_": "diverging" },
			"length": 20.0
		},
		{
			"from_": { "switch": "A", "type_": "approach" },
			"to": { "switch": "B", "type_": "through" },
			"length": 30.0
		}
	],
	"trains": [
		{
			"tag": "T",
			"speed": 1.0,
			"head_progress": 0.5,
			"length": 15.0,
			"history": [
				{ "switch": "A", "type_": "through" },
				{ "switch": "B", "type_": "through" }
			]
		}
	]
}
```

Notes:
- `type_` must be one of: `approach`, `through`, `diverging`
- A track connects two branch endpoints (`from_` and `to`) and has a `length`
- A train’s `history` is ordered **head → tail** (`history[0]` is the head branch, `history[-1]` is the tail branch)

---

## Graph encoding (brief)

`System.encode()` produces a `networkx.DiGraph` with numeric node ids and feature vectors:
- node features live on `node["x"]`
- edge features live on `edge["x"]`

This is useful if you want to compile the system into a graph representation for ML workflows.
For PyTorch Geometric, a typical path is to convert the NetworkX graph into a `torch_geometric.data.Data`:

```python
from torch_geometric.utils import from_networkx

nx_g = system.encode(edge_subdivisions=10)
data = from_networkx(nx_g, group_node_attrs=["x"], group_edge_attrs=["x"])
```

---

## Project status

This is intentionally small and pragmatic: it’s a compact foundation for experimenting with dense layouts,
simulation stepping, and graph encodings—grown out of a real-world model train setup.
