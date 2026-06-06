# Architecture Diagram

High-level component architecture of `simgen`.

```mermaid
flowchart TB
    subgraph Interface
        CLI["cli.py (simgen command)"]
        API["Python API (simgen.*)"]
    end

    subgraph Interpretation["Interpretation layer (simgen.llm)"]
        GEN["generator.SimulationGenerator"]
        CLIENT["client.ClaudeClient"]
        SCHEMA["schema (JSON schema + parser)"]
        PROMPTS["prompts"]
    end

    subgraph Core["Scientific core"]
        SPEC["simulations.spec.PhenomenonSpec"]
        BASE["simulations.base (compiler + ODESimulation)"]
        RUN["simulations.runner.run_spec"]
        NUM["numerical_methods (integrate, solvers, linalg)"]
        MATH["mathematics (analysis)"]
        PHYS["physics (constants, units)"]
        LIB["simulations.library (5 reference specs)"]
    end

    subgraph Presentation["Generation & presentation"]
        CODEGEN["codegen.renderer + templates"]
        PLOT["visualization.plotting"]
        SCENE["scenes (Manim helpers)"]
        REPORT["reporting.latex"]
    end

    CLI --> GEN
    API --> GEN
    GEN --> CLIENT --> PROMPTS
    CLIENT --> SCHEMA --> SPEC
    GEN -. fallback .-> LIB --> SPEC
    SPEC --> CODEGEN
    SPEC --> BASE --> RUN --> NUM
    RUN --> MATH
    BASE --> PHYS
    CODEGEN --> PLOT
    CODEGEN --> SCENE
    CODEGEN --> REPORT
```

## Layers

- **Interface** — the `simgen` CLI and the importable Python API.
- **Interpretation** — converts prompts to a validated `PhenomenonSpec` (Claude
  API), with the offline library as a fallback.
- **Scientific core** — the spec model, the spec→simulation compiler, the
  integrators, and the analysis/constants utilities.
- **Generation & presentation** — renders the spec into standalone code, plots,
  Manim scenes and LaTeX reports.

The `PhenomenonSpec` is the contract that every layer shares, guaranteeing that
all artifacts describe the same model.
