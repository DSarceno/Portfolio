# Data Flow Diagram

How data moves from a prompt to finished artifacts.

```mermaid
sequenceDiagram
    actor User
    participant CLI as simgen CLI / API
    participant Gen as SimulationGenerator
    participant Claude as ClaudeClient (Anthropic)
    participant Schema as schema.spec_from_json
    participant Lib as library (fallback)
    participant Code as codegen.renderer
    participant Run as runner.run_spec
    participant Plot as visualization.plotting

    User->>CLI: prompt or --from-library slug
    alt prompt + API available
        CLI->>Gen: spec_from_prompt(prompt)
        Gen->>Claude: generate_spec_dict(prompt) [tool use]
        Claude-->>Gen: spec dict
        Gen->>Schema: validate + parse
        Schema-->>Gen: PhenomenonSpec
    else no key / error / --from-library
        CLI->>Lib: get_spec(slug) or keyword match
        Lib-->>Gen: PhenomenonSpec
    end
    Gen->>Code: generate_project(spec)
    Code-->>User: simulation.py, plot.py, scene.py, report.tex, README.md, spec.json
    opt --run / --plot
        CLI->>Run: run_spec(spec)
        Run-->>CLI: SimulationResult (t, states)
        CLI->>Plot: render_all_plots(spec, result)
        Plot-->>User: figure_*.png
    end
```

## Key data structures

| Stage | Data |
|-------|------|
| Prompt | free text |
| Model output | JSON dict matching `SPEC_JSON_SCHEMA` |
| Parsed | `PhenomenonSpec` (validated) |
| Generated | files on disk + `spec.json` |
| Run | `SimulationResult` (`t`, `states`, `symbols`) |
| Plotted | PNG figures |

The `PhenomenonSpec` is serialised to `spec.json` in every generated project, so
the exact inputs are always recoverable and the pipeline is reproducible.
