@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM  run_all.bat  -  one-shot full pipeline for the simgen project.
REM
REM  Steps:
REM    1. Activate the virtual environment (simu\).
REM    2. Ensure the simgen package is installed (editable).
REM    3. Generate a project from a prompt or a library phenomenon.
REM    4. Run the simulation and render the plots (--run --plot).
REM    5. Render the Manim animation        (only if manim is on PATH).
REM    6. Compile the LaTeX report          (only if pdflatex is on PATH).
REM
REM  Usage:
REM    run_all.bat                        Default demo (lorenz-system, offline)
REM    run_all.bat -l ^<slug^>             Generate from a built-in library phenomenon
REM    run_all.bat "^<prompt words...^>"   Generate from a natural-language prompt (needs API key)
REM
REM  Examples:
REM    run_all.bat
REM    run_all.bat -l double-pendulum
REM    run_all.bat "The Lorenz attractor with sigma=10, rho=28, beta=8/3"
REM ============================================================================

REM --- Always operate from the project root (this script's folder) -----------
cd /d "%~dp0"

set "VENV=simu\Scripts\activate.bat"
set "OUTDIR=generated"

REM --- 1. Activate the virtual environment -----------------------------------
if not exist "%VENV%" (
    echo [ERROR] Virtual environment not found at "%VENV%".
    echo         Create it first, e.g.:  python -m venv simu
    exit /b 1
)
call "%VENV%"

REM --- 2. Ensure the simgen package is installed -----------------------------
python -c "import simgen" 1>nul 2>nul
if errorlevel 1 (
    echo [setup] Installing simgen in editable mode...
    pip install -e ".[dev]"
    if errorlevel 1 (
        echo [ERROR] Failed to install simgen.
        exit /b 1
    )
)

REM --- 3. Decide the generation mode from the arguments ----------------------
if "%~1"=="" (
    echo [info] No argument given - using default library demo: lorenz-system
    set "GEN_ARGS=--from-library lorenz-system"
) else if /i "%~1"=="-l" (
    if "%~2"=="" (
        echo [ERROR] -l requires a library slug, e.g.:  run_all.bat -l double-pendulum
        exit /b 2
    )
    set "GEN_ARGS=--from-library %~2"
) else (
    REM Treat the whole quoted command line as a natural-language prompt.
    set "GEN_ARGS=%*"
)

REM --- 4. Generate + run + plot ----------------------------------------------
echo.
echo [step] Generating project...
simgen generate !GEN_ARGS! -o "%OUTDIR%" --run --plot
if errorlevel 1 (
    echo [ERROR] Generation failed.
    exit /b 1
)

REM --- 5. Locate the most recently generated project ------------------------
set "PROJDIR="
for /f "delims=" %%d in ('dir /b /ad /o-d "%OUTDIR%" 2^>nul') do (
    set "PROJDIR=%OUTDIR%\%%d"
    goto :found_proj
)
:found_proj
if not defined PROJDIR (
    echo [ERROR] Could not locate a generated project under "%OUTDIR%".
    exit /b 1
)
echo [info] Project directory: !PROJDIR!

REM --- 6. Render the Manim animation (needs both manim and ffmpeg) ----------
where manim 1>nul 2>nul
set "HAVE_MANIM=%errorlevel%"
where ffmpeg 1>nul 2>nul
set "HAVE_FFMPEG=%errorlevel%"
if not "%HAVE_MANIM%"=="0" (
    echo [skip] Manim not found on PATH - skipping animation render.
) else if not "%HAVE_FFMPEG%"=="0" (
    echo [skip] FFmpeg not found on PATH - skipping animation render ^(Manim needs it to encode^).
) else (
    echo.
    echo [step] Rendering Manim animation...
    manim -qh -a "!PROJDIR!\scene.py"
    if errorlevel 1 echo [warn] Manim render reported an error.
)

REM --- 7. Compile the LaTeX report (only if pdflatex is available) ----------
where pdflatex 1>nul 2>nul
if errorlevel 1 (
    echo [skip] pdflatex not found on PATH - skipping report compilation.
) else (
    echo.
    echo [step] Compiling LaTeX report...
    pushd "!PROJDIR!"
    pdflatex -interaction=nonstopmode report.tex 1>nul 2>nul
    pdflatex -interaction=nonstopmode report.tex 1>nul 2>nul
    popd
    echo [info] Report compiled: !PROJDIR!\report.pdf
)

echo.
echo [done] Full pipeline complete for: !PROJDIR!
endlocal
