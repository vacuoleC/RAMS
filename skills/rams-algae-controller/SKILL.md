---
name: rams-algae-controller
description: Control RAMS-Algae-Monitoring through its frozen governance and deterministic supervised runtime. Use when resuming, executing, reviewing, pausing, recovering, or completing this specific project.
---

# RAMS-Algae-Monitoring Controller

This is the only project Controller Skill for project `47077f0c-1772-47c8-ae2a-b99a38928335`, Controller revision `1`.

## Reconstruct before acting

Use the host recovery payload when present. Otherwise locate the project through the local supervised registry. Verify `controller_manifest.json`, then load only the context needed from `project_design.yaml`, `monitor.yaml`, the current Module design, and referenced ledger evidence.

Call the supervised runtime for every state transition, approval, guard, lease, ledger, budget, installation-integrity, and recovery operation. Treat runtime or guard indeterminacy as fail-closed.

## Execute within frozen scope

Choose business work only inside the frozen Module graph and current project state. Keep Module, Event, and Subtask identities as project data. Refine later Subtasks only inside an existing Module and without inventing unavailable inputs. Require a change request for scope, Module graph, policy, required-capability, or managed Controller changes.

Continue through safe executable work until the project reaches `blocked`, `paused`, `review`, `complete`, `cancelled`, or a genuine host/time limit. Preserve evidence and leave an exact recovery point before yielding at a host limit.

## Hard boundaries

- Do not write monitor.yaml.
- Do not append whatwedo.jsonl.
- Do not execute WAL recovery directly.
- Do not modify this Controller Skill or other generator-owned files.
- Do not create Module or leaf Skills.
- Do not invoke the Meta Skill implicitly.
- Do not treat adapter installation as activation; require the formal local capability report.
