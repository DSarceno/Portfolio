@echo off
REM ============================================================
REM  FIFA World Cup 2026 Quiniela Predictor V2 - Full pipeline
REM ------------------------------------------------------------
REM  Orquesta: ingesta -> ratings -> features -> entrenamiento
REM           -> predicciones -> picks -> simulacion -> reportes
REM ============================================================

setlocal enabledelayedexpansion
cd /d %~dp0

set "START_TIME=%TIME%"
set "N_RUNS=2000"
set "HISTORICAL_START_YEAR=2014"
set "KAGGLE_CSV=data\raw\kaggle\results.csv"

echo.
echo ============================================================
echo  FIFA World Cup 2026 Quiniela Predictor V2 - FULL PIPELINE
echo  Inicio: %DATE% %TIME%
echo ============================================================

REM ------------------------------------------------------------
REM  Setup: activar venv (intenta ..\worldCup, luego .venv)
REM ------------------------------------------------------------
echo.
echo [setup] Activando entorno virtual...
if exist "..\worldCup\Scripts\activate.bat" (
    call "..\worldCup\Scripts\activate.bat"
    echo [setup]   -^> ..\worldCup activado
) else if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
    echo [setup]   -^> .venv activado
) else (
    echo [setup]   -^> Sin venv detectado; usando python del sistema
)

python --version >nul 2>nul
if errorlevel 1 (
    echo [setup] ERROR: python no encontrado en PATH
    goto :failure
)

REM ------------------------------------------------------------
REM  Paso 1: Importar histórico Kaggle (si results.csv existe)
REM ------------------------------------------------------------
echo.
echo ============================================================
echo  [1/9] Importando histórico Kaggle (anios %HISTORICAL_START_YEAR%-2026)
echo ============================================================
if exist "%KAGGLE_CSV%" (
    python scripts\import_kaggle_history.py --csv "%KAGGLE_CSV%" --start-year %HISTORICAL_START_YEAR% --replace
    if errorlevel 1 (
        set "FAIL_STEP=1 - Kaggle import"
        goto :failure
    )
) else (
    echo [1/9] AVISO: %KAGGLE_CSV% no encontrado.
    echo [1/9] Descargalo de https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
    echo [1/9] y colocalo en %KAGGLE_CSV%
    echo [1/9] Continuando con lo que haya en data\interim\matches_unified.csv...
)

REM ------------------------------------------------------------
REM  Paso 2: Fixtures WC 2026 desde football-data.org
REM ------------------------------------------------------------
echo.
echo ============================================================
echo  [2/9] Descargando fixtures WC 2026 desde football-data.org
echo ============================================================
python scripts\bootstrap_historical_data.py --competitions WC --start-year 2026 --end-year 2026 --no-kaggle
if errorlevel 1 (
    set "FAIL_STEP=2 - Bootstrap WC 2026"
    goto :failure
)

REM ------------------------------------------------------------
REM  Paso 3: Ratings (Elo + PI + Form)
REM ------------------------------------------------------------
echo.
echo ============================================================
echo  [3/9] Construyendo ratings pre-torneo
echo ============================================================
python scripts\build_ratings.py
if errorlevel 1 (
    set "FAIL_STEP=3 - Build ratings"
    goto :failure
)

REM ------------------------------------------------------------
REM  Paso 4: Feature matrix
REM ------------------------------------------------------------
echo.
echo ============================================================
echo  [4/9] Construyendo feature matrix
echo ============================================================
python scripts\run_pipeline.py
if errorlevel 1 (
    set "FAIL_STEP=4 - Run pipeline"
    goto :failure
)

REM ------------------------------------------------------------
REM  Paso 5: Entrenamiento (multinomial + XGBoost + Poisson + calibrator)
REM ------------------------------------------------------------
echo.
echo ============================================================
echo  [5/9] Entrenando modelos
echo ============================================================
python scripts\train_models.py
if errorlevel 1 (
    set "FAIL_STEP=5 - Train models"
    goto :failure
)

REM ------------------------------------------------------------
REM  Paso 6: Predicciones fase de grupos
REM ------------------------------------------------------------
echo.
echo ============================================================
echo  [6/9] Prediciendo fase de grupos
echo ============================================================
python scripts\predict_group_stage.py
if errorlevel 1 (
    set "FAIL_STEP=6 - Predict group stage"
    goto :failure
)

REM ------------------------------------------------------------
REM  Paso 7: Exportar quinielas (4 perfiles de riesgo)
REM ------------------------------------------------------------
echo.
echo ============================================================
echo  [7/9] Exportando quinielas (safe / balanced / aggressive / contrarian)
echo ============================================================
python scripts\export_quiniela_sheet.py
if errorlevel 1 (
    set "FAIL_STEP=7 - Export quiniela sheets"
    goto :failure
)

REM ------------------------------------------------------------
REM  Paso 8: Simulacion Monte-Carlo del torneo
REM ------------------------------------------------------------
echo.
echo ============================================================
echo  [8/9] Simulando torneo Monte-Carlo (n_runs=%N_RUNS%)
echo ============================================================
python scripts\simulate_tournament.py --n-runs %N_RUNS%
if errorlevel 1 (
    set "FAIL_STEP=8 - Simulate tournament"
    goto :failure
)

REM ------------------------------------------------------------
REM  Paso 9: Compilar reportes LaTeX a PDF
REM ------------------------------------------------------------
echo.
echo ============================================================
echo  [9/9] Compilando reportes LaTeX a PDF
echo ============================================================

where pdflatex >nul 2>nul
if errorlevel 1 (
    echo [9/9] AVISO: pdflatex no encontrado en PATH.
    echo [9/9] Instala TeX Live ^(https://www.tug.org/texlive/^) o MiKTeX ^(https://miktex.org/^).
    echo [9/9] Saltando compilacion PDF.
    goto :skip_latex
)

echo.
echo [9/9] Compilando reports\academic\main.tex ...
pushd reports\academic
pdflatex -interaction=nonstopmode -halt-on-error main.tex >nul 2>nul
pdflatex -interaction=nonstopmode -halt-on-error main.tex >nul 2>nul
if exist main.pdf (
    echo [9/9]   -^> reports\academic\main.pdf generado
) else (
    echo [9/9]   -^> ERROR: main.pdf no se genero. Revisa reports\academic\main.log
)
popd

echo.
echo [9/9] Compilando reports\dashboard\main.tex ...
pushd reports\dashboard
pdflatex -interaction=nonstopmode -halt-on-error main.tex >nul 2>nul
pdflatex -interaction=nonstopmode -halt-on-error main.tex >nul 2>nul
if exist main.pdf (
    echo [9/9]   -^> reports\dashboard\main.pdf generado
) else (
    echo [9/9]   -^> ERROR: main.pdf no se genero. Revisa reports\dashboard\main.log
)
popd

:skip_latex

REM ------------------------------------------------------------
REM  Resumen final
REM ------------------------------------------------------------
echo.
echo ============================================================
echo  PIPELINE COMPLETADO EXITOSAMENTE
echo  Inicio: %START_TIME%
echo  Fin:    %TIME%
echo ============================================================
echo.
echo Outputs generados:
echo   outputs\predictions\group_stage_predictions.csv
echo   outputs\picks\quiniela_safe.csv
echo   outputs\picks\quiniela_balanced.csv
echo   outputs\picks\quiniela_aggressive.csv
echo   outputs\picks\quiniela_contrarian.csv
echo   outputs\simulations\championship_probabilities.csv
echo   outputs\simulations\tournament_probabilities.csv
echo   outputs\simulations\round_reached_probabilities.csv
echo   outputs\simulations\bracket_paths.csv
echo   outputs\diagnostics\composite_ratings.csv
echo   outputs\diagnostics\feature_importance.csv
echo.
if exist "reports\academic\main.pdf" (
    echo Reportes PDF:
    echo   reports\academic\main.pdf
)
if exist "reports\dashboard\main.pdf" (
    echo   reports\dashboard\main.pdf
)
echo.

endlocal
exit /b 0

:failure
echo.
echo ============================================================
echo  PIPELINE FALLO en el paso: %FAIL_STEP%
echo  Errorlevel: %errorlevel%
echo ============================================================
echo Revisa los logs en la carpeta logs\ para mas detalles.
endlocal
exit /b 1
