"""Tests for the spec model, simulation core, library and code generation."""

from __future__ import annotations

import py_compile

import numpy as np
import pytest

from simgen import generate_project, run_spec
from simgen.simulations.library import all_specs, get_spec, list_specs
from simgen.simulations.spec import (
    Parameter,
    PhenomenonSpec,
    SpecValidationError,
    StateVariable,
)


def _minimal_spec() -> PhenomenonSpec:
    return PhenomenonSpec(
        name="Exponential Decay",
        summary="A single linearly decaying variable.",
        state_variables=[StateVariable("x", "amount")],
        parameters=[Parameter("k", 1.0, "decay rate")],
        derivatives=["-k * x"],
        initial_conditions=[1.0],
        t_start=0.0,
        t_end=5.0,
        num_points=500,
        method="rk4",
    )


def test_spec_validation_mismatched_derivatives() -> None:
    with pytest.raises(SpecValidationError):
        PhenomenonSpec(
            name="bad",
            summary="bad",
            state_variables=[StateVariable("x")],
            derivatives=["x", "y"],  # too many
            initial_conditions=[1.0],
        )


def test_spec_validation_state_param_overlap() -> None:
    with pytest.raises(SpecValidationError):
        PhenomenonSpec(
            name="bad",
            summary="bad",
            state_variables=[StateVariable("x")],
            parameters=[Parameter("x", 1.0)],
            derivatives=["-x"],
            initial_conditions=[1.0],
        )


def test_spec_roundtrip_to_from_dict() -> None:
    spec = _minimal_spec()
    restored = PhenomenonSpec.from_dict(spec.to_dict())
    assert restored.slug == spec.slug
    assert restored.derivatives == spec.derivatives
    assert restored.state_symbols == spec.state_symbols


def test_spec_slug_and_class_name() -> None:
    spec = _minimal_spec()
    assert spec.slug == "exponential-decay"
    assert spec.class_name == "ExponentialDecay"


def test_exponential_decay_matches_analytic() -> None:
    spec = _minimal_spec()
    result = run_spec(spec)
    analytic = np.exp(-result.t)
    assert np.max(np.abs(result.column("x") - analytic)) < 1e-4


def test_result_column_lookup() -> None:
    result = run_spec(_minimal_spec())
    assert np.allclose(result.column("t"), result.t)
    with pytest.raises(KeyError):
        result.column("does_not_exist")


@pytest.mark.parametrize("slug", list_specs())
def test_library_specs_run(slug: str) -> None:
    """Every reference spec integrates to a finite trajectory."""
    result = run_spec(get_spec(slug))
    assert result.success
    assert np.all(np.isfinite(result.states))
    assert result.states.shape == (result.n_steps, get_spec(slug).dimension)


def test_all_specs_validate() -> None:
    for spec in all_specs():
        spec.validate()  # must not raise


def test_generate_project_creates_files(tmp_path) -> None:
    spec = get_spec("lorenz-system")
    project = generate_project(spec, tmp_path)
    for path in (
        project.simulation_path,
        project.plot_path,
        project.scene_path,
        project.report_path,
        project.readme_path,
        project.spec_path,
    ):
        assert path.is_file()
    # Generated Python must be syntactically valid.
    py_compile.compile(str(project.simulation_path), doraise=True)
    py_compile.compile(str(project.plot_path), doraise=True)


def test_generated_simulation_is_importable_and_correct(tmp_path) -> None:
    """The generated standalone simulation reproduces the analytic decay."""
    import importlib.util

    project = generate_project(_minimal_spec(), tmp_path)
    spec_loader = importlib.util.spec_from_file_location(
        "generated_sim", project.simulation_path
    )
    assert spec_loader and spec_loader.loader
    module = importlib.util.module_from_spec(spec_loader)
    spec_loader.loader.exec_module(module)
    t, states = module.simulate()
    assert np.max(np.abs(states[:, 0] - np.exp(-t))) < 1e-4
