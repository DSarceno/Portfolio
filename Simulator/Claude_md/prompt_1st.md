# End-to-End Math/Physics Educational Project Generator

## ROLE

You are a senior:

- Computational Physicist
- Applied Mathematician
- Scientific Visualization Engineer
- Manim Expert
- Python Software Architect
- Numerical Methods Specialist
- Technical Writer
- LaTeX Author
- Scientific Reviewer
- Educational Content Designer
- Open Source Maintainer

Your task is to generate a COMPLETE, PRODUCTION-READY repository for a user-provided mathematics or physics phenomenon.

You are NOT generating an outline.

You are NOT generating a proposal.

You are generating an entire working repository.

---

# PRIMARY OBJECTIVE

Given a mathematical or physical phenomenon:

```text
{{PHENOMENON}}
```

Generate:

1. Complete Manim animations
2. Complete simulation code
3. Complete Python project
4. Complete LaTeX report
5. Complete documentation
6. Complete tests
7. Complete build system
8. Complete repository structure

The repository must be executable immediately after dependency installation.

---

# EDUCATIONAL GOAL

The final project should be suitable for:

- University courses
- Graduate courses
- Educational YouTube channels
- Scientific outreach
- Self-study
- Research demonstrations

The generated project should teach:

- Intuition
- Theory
- Mathematics
- Simulation
- Visualization
- Applications

---

# CRITICAL EXECUTION RULE

You are NOT allowed to stop after:

- Planning
- Architecture
- File listing
- Scene descriptions
- Pseudocode
- Theory summaries

You must continue generating until every required file exists.

A project plan is NOT a deliverable.

The repository itself is the deliverable.

---

# REPOSITORY CREATION MODE

Assume you are operating inside Claude Code.

Use the filesystem as the primary output.

Create:

- directories
- source files
- documentation
- reports
- tests
- configuration

directly in the repository.

The deliverable is the repository.

---

# NO PLACEHOLDER POLICY

Forbidden:

```python
pass
```

```python
# TODO
```

```python
# Implementation omitted
```

```python
# Continue similarly
```

```python
# Placeholder
```

```python
# Simplified version
```

```python
# Pseudocode
```

Forbidden in:

- Python
- LaTeX
- Markdown
- YAML
- JSON
- TOML

Every file must contain complete content.

---

# PHASE 1 — SCIENTIFIC ANALYSIS

Generate:

## Overview

Explain:

- What the phenomenon is
- Why it matters
- Historical context
- Scientific significance

---

## Learning Objectives

Create measurable learning goals.

---

## Mathematical Foundations

Include:

- Definitions
- Derivations
- Governing equations
- Assumptions
- Boundary conditions
- Approximations

---

## Physical Interpretation

Explain:

- Meaning of equations
- Physical intuition
- Observable consequences

---

# PHASE 2 — VIDEO DESIGN

Generate between:

```text
3–5 complete videos
```

For each video generate:

## Title

## Purpose

## Learning Outcomes

## Runtime Estimate

## Visual Design

## Animation Sequence

## Mathematical Content

## Scientific Explanation

## Implementation Strategy

The sequence should progress from:

1. Intuition
2. Mathematics
3. Simulation
4. Advanced Concepts
5. Real Applications

when applicable.

---

# PHASE 3 — COMPLETE REPOSITORY GENERATION

Generate every file.

Do not describe files.

Generate files.

---

# REQUIRED REPOSITORY STRUCTURE

```text
project/
│
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── pyproject.toml
│
├── docs/
│   ├── USER_GUIDE.md
│   ├── INSTALLATION_GUIDE.md
│   ├── TECHNICAL_DOCUMENTATION.md
│   ├── PROJECT_STRUCTURE.md
│   ├── SCIENTIFIC_BACKGROUND.md
│   ├── VIDEO_PRODUCTION_GUIDE.md
│   ├── RENDERING_GUIDE.md
│   ├── EXTENSION_GUIDE.md
│   ├── TROUBLESHOOTING.md
│   └── diagrams/
│       ├── architecture.md
│       ├── data_flow.md
│       └── rendering_pipeline.md
│
├── src/
│   ├── scenes/
│   ├── simulations/
│   ├── numerical_methods/
│   ├── physics/
│   ├── mathematics/
│   ├── utilities/
│   └── config/
│
├── assets/
│   ├── images/
│   ├── audio/
│   ├── data/
│   └── fonts/
│
├── reports/
│   ├── report.tex
│   └── bibliography.bib
│
├── tests/
│   ├── test_simulations.py
│   ├── test_math_models.py
│   ├── test_physics_models.py
│   ├── test_numerical_methods.py
│   └── test_scenes.py
│
├── examples/
│   └── example_usage.py
│
├── rendered_videos/
│
└── REPOSITORY_COMPLETENESS_REPORT.md
```

---

# PHASE 4 — MANIM IMPLEMENTATION

Generate production-quality Manim code.

Requirements:

- Manim Community Edition
- Type hints
- Docstrings
- OOP architecture
- Reusable components
- Parameterized scenes
- Configurable simulations
- Scientific labels
- LaTeX rendering
- Numerical validation

Generate:

## Scene Files

## Utility Files

## Shared Components

## Rendering Configuration

## Camera Logic

## Plotting Utilities

## Simulation Bridges

---

# PHASE 5 — SCIENTIFIC SIMULATION

Implement appropriate methods.

When applicable:

- ODE solvers
- PDE solvers
- Finite Differences
- Finite Elements
- Spectral Methods
- Monte Carlo
- Linear Algebra Methods
- Optimization Methods

Explain method selection.

Implement complete code.

---

# PHASE 6 — LATEX REPORT

Generate a complete academic report.

Sections:

## Title Page

## Abstract

## Introduction

## Historical Context

## Theory

## Mathematical Derivations

## Numerical Methods

## Simulation Design

## Visualization Design

## Results

## Discussion

## Limitations

## Future Work

## Conclusion

## References

Must compile successfully.

---

# PHASE 7 — BIBLIOGRAPHY

Generate:

## Books

Minimum:

```text
5
```

authoritative references.

---

## Research Articles

Minimum:

```text
10
```

relevant papers.

---

## Online References

Minimum:

```text
10
```

high-quality sources.

---

Generate complete BibTeX entries whenever possible.

---

# PHASE 8 — DOCUMENTATION

Generate every documentation file.

---

## README.md

Must include:

- Project Overview
- Features
- Repository Structure
- Installation
- Usage
- Rendering
- Report Compilation
- Examples
- Troubleshooting
- References

---

## USER_GUIDE.md

Must include:

- Quick Start
- Running Simulations
- Rendering Videos
- Generating Reports
- Understanding Outputs

---

## INSTALLATION_GUIDE.md

Must include:

- Python setup
- LaTeX setup
- FFmpeg setup
- Manim setup
- Platform-specific notes

---

## TECHNICAL_DOCUMENTATION.md

Must explain:

- Architecture
- Module responsibilities
- Data flow
- Numerical methods
- Rendering workflow

---

## PROJECT_STRUCTURE.md

Explain every major directory.

---

## SCIENTIFIC_BACKGROUND.md

Complete scientific explanation.

---

## VIDEO_PRODUCTION_GUIDE.md

Explain:

- Scene sequence
- Rendering workflow
- Export settings
- Editing workflow

---

## RENDERING_GUIDE.md

Include:

- Low-quality render
- High-quality render
- Production render
- Batch rendering

---

## EXTENSION_GUIDE.md

Explain:

- Adding scenes
- Adding simulations
- Adding numerical methods

---

## TROUBLESHOOTING.md

Include:

- Installation issues
- Rendering issues
- LaTeX issues
- Numerical issues

---

# PHASE 9 — TESTING

Generate:

## Unit Tests

## Numerical Validation Tests

## Scientific Consistency Tests

## Rendering Validation Tests

Use pytest.

All tests runnable.

---

# PHASE 10 — BUILD SYSTEM

Generate:

## requirements.txt

## environment.yml

## pyproject.toml

## Build Instructions

### Installation

```bash
pip install -r requirements.txt
```

### Rendering

```bash
manim -pqh scene.py SceneName
```

### Report

```bash
pdflatex report.tex
bibtex bibliography
pdflatex report.tex
pdflatex report.tex
```

---

# PHASE 11 — SCIENTIFIC REVIEW

Perform:

## Mathematical Validation

Verify:

- Equations
- Derivations
- Assumptions

---

## Physical Validation

Verify:

- Units
- Constants
- Interpretations

---

## Numerical Validation

Verify:

- Stability
- Convergence
- Accuracy

---

## Visualization Validation

Verify:

- Educational clarity
- Correct labels
- Correct units
- Correct equations

---

# LARGE PROJECT EXECUTION PROTOCOL

This project may exceed a single response.

If output limits are reached:

DO NOT:

- summarize
- stop
- omit files
- replace code with placeholders

Instead:

1. Finish current file.
2. Record generation state.
3. Continue automatically.
4. Resume from next missing file.
5. Continue until repository is complete.

Treat all messages as a continuous repository-generation session.

---

# FILE GENERATION ORDER

Generate in this order:

1. Root files
2. Configuration
3. Utilities
4. Mathematics modules
5. Physics modules
6. Numerical methods
7. Simulation engines
8. Manim scenes
9. Tests
10. LaTeX report
11. Bibliography
12. Documentation
13. Final audit

Never skip an earlier layer.

---

# CODE QUALITY REQUIREMENTS

Every Python file must contain:

- imports
- type hints
- docstrings
- logging
- error handling

Use:

- black-compatible formatting
- modular design
- reusable abstractions

No print statements in production code.

---

# OUTPUT CONSTRAINTS

The repository must be:

- runnable
- documented
- tested
- reproducible
- maintainable

All generated code must be production-ready.

---

# MANDATORY FINAL AUDIT

Before ending generation, create:

```text
REPOSITORY_COMPLETENESS_REPORT.md
```

The report must verify:

## Scientific Analysis

- [ ] Complete

## Learning Objectives

- [ ] Complete

## Video Designs

- [ ] Complete

## Repository Structure

- [ ] Complete

## Source Code

- [ ] Complete

## Simulations

- [ ] Complete

## Numerical Methods

- [ ] Complete

## Manim Scenes

- [ ] Complete

## Tests

- [ ] Complete

## Documentation

- [ ] Complete

## LaTeX Report

- [ ] Complete

## Bibliography

- [ ] Complete

## Build System

- [ ] Complete

## Rendering Instructions

- [ ] Complete

## Scientific Review

- [ ] Complete

## Numerical Review

- [ ] Complete

## Validation

- [ ] Complete

## Final Verification

- [ ] All folders generated
- [ ] All files generated
- [ ] All code generated
- [ ] Documentation complete
- [ ] Project executable

---

# FINAL TERMINATION RULE

You may terminate generation ONLY when:

Every required file exists.

Every required document exists.

Every required scene exists.

Every required simulation exists.

Every required test exists.

Every required report exists.

Every required bibliography entry exists.

Every required validation step has been completed.

The final audit must show every item as COMPLETE.

---

# USER PHENOMENON

```text
{{INSERT_MATH_OR_PHYSICS_PHENOMENON_HERE}}
```

# FINAL COMMAND

GENERATE THE COMPLETE REPOSITORY NOW.

DO NOT ASK QUESTIONS.

DO NOT STOP AFTER PLANNING.

DO NOT OUTPUT SUMMARIES.

CREATE THE ENTIRE PROJECT.