@echo off
REM ============================================================
REM  FIFA World Cup 2026 Quiniela Predictor V2 - Full pipeline
REM ------------------------------------------------------------
REM  Orquesta: ingesta -> ratings -> valor de plantilla -> features
REM           -> entrenamiento -> predicciones -> picks -> simulacion
REM           -> reportes
REM ============================================================

setlocal enabledelayedexpansion
cd /d %~dp0

set "START_TIME=%TIME%"
set "N_RUNS=2000"
set "HISTORICAL_START_YEAR=2014"
set "KAGGLE_CSV=data\raw\kaggle\results.csv"
set "PLAYER_SCORES=data\raw\kaggle\players-scores\player_valuations.csv"

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
echo  [1/10] Importando histórico Kaggle (anios %HISTORICAL_START_YEAR%-2026)
echo ============================================================
if exist "%KAGGLE_CSV%" (
    python scripts\import_kaggle_history.py --csv "%KAGGLE_CSV%" --start-year %HISTORICAL_START_YEAR% --replace
    if errorlevel 1 (
        set "FAIL_STEP=1 - Kaggle import"
        goto :failure
    )
) else (
    echo [1/10] AVISO: %KAGGLE_CSV% no encontrado.
    echo [1/10] Descargalo de https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
    echo [1/10] y colocalo en %KAGGLE_CSV%
    echo [1/10] Continuando con lo que haya en data\interim\matches_unified.csv...
)

REM ------------------------------------------------------------
REM  Paso 2: Fixtures WC 2026 desde football-data.org
REM ------------------------------------------------------------
echo.
echo ============================================================
echo  [2/10] Descargando fixtures WC 2026 desde football-data.org
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
echo  [3/10] Construyendo ratings pre-torneo
echo ============================================================
python scripts\build_ratings.py
if errorlevel 1 (
    set "FAIL_STEP=3 - Build ratings"
    goto :failure
)

REM ------------------------------------------------------------
REM  Paso 4: Valor de plantilla (A.2) - opcional, degrada limpio
REM ------------------------------------------------------------
echo.
echo ============================================================
echo  [4/10] Construyendo snapshots de valor de plantilla (A.2)
echo ============================================================
if not exist "%PLAYER_SCORES%" goto :skip_squad
python scripts\build_squad_values.py
if errorlevel 1 echo [4/10] AVISO: build_squad_values fallo; continuando sin valor de plantilla.
goto :after_squad
:skip_squad
echo [4/10] AVISO: %PLAYER_SCORES% no encontrado.
echo [4/10] Descarga el dataset Kaggle davidcariboo/player-scores a
echo [4/10] data\raw\kaggle\players-scores\ para activar la feature de valor de plantilla.
echo [4/10] Continuando sin ella; el pipeline degrada limpio.
:after_squad

REM ------------------------------------------------------------
REM  Paso 5: Feature matrix
REM ------------------------------------------------------------
echo.
echo ============================================================
echo  [5/10] Construyendo feature matrix
echo ============================================================
python scripts\run_pipeline.py
if errorlevel 1 (
    set "FAIL_STEP=5 - Run pipeline"
    goto :failure
)

REM ------------------------------------------------------------
REM  Paso 6: Entrenamiento (multinomial + XGBoost + Poisson + calibrator)
REM ------------------------------------------------------------
echo.
echo ============================================================
echo  [6/10] Entrenando modelos
echo ============================================================
python scripts\train_models.py
if errorlevel 1 (
    set "FAIL_STEP=6 - Train models"
    goto :failure
)

REM ------------------------------------------------------------
REM  Paso 7: Predicciones (H/D/A + marcadores mas probables)
REM ------------------------------------------------------------
echo.
echo ============================================================
echo  [7/10] Prediciendo partidos: H/D/A + marcadores
echo ============================================================
python scripts\predict_group_stage.py
if errorlevel 1 (
    set "FAIL_STEP=7 - Predict group stage"
    goto :failure
)
python scripts\predict_scorelines.py
if errorlevel 1 (
    set "FAIL_STEP=7 - Predict scorelines"
    goto :failure
)

REM ------------------------------------------------------------
REM  Paso 8: Exportar quinielas (4 perfiles de riesgo)
REM ------------------------------------------------------------
echo.
echo ============================================================
echo  [8/10] Exportando quinielas (safe / balanced / aggressive / contrarian)
echo ============================================================
python scripts\export_quiniela_sheet.py
if errorlevel 1 (
    set "FAIL_STEP=8 - Export quiniela sheets"
    goto :failure
)

REM ------------------------------------------------------------
REM  Paso 9: Simulacion Monte-Carlo del torneo
REM ------------------------------------------------------------
echo.
echo ============================================================
echo  [9/10] Simulando torneo Monte-Carlo (n_runs=%N_RUNS%)
echo ============================================================
python scripts\simulate_tournament.py --n-runs %N_RUNS%
if errorlevel 1 (
    set "FAIL_STEP=9 - Simulate tournament"
    goto :failure
)

REM ------------------------------------------------------------
REM  Paso 10: Compilar reportes LaTeX a PDF
REM ------------------------------------------------------------
echo.
echo ============================================================
echo  [10/10] Compilando reportes LaTeX a PDF
echo ============================================================

where pdflatex >nul 2>nul
if errorlevel 1 (
    echo [10/10] AVISO: pdflatex no encontrado en PATH.
    echo [10/10] Instala TeX Live ^(https://www.tug.org/texlive/^) o MiKTeX ^(https://miktex.org/^).
    echo [10/10] Saltando compilacion PDF.
    goto :skip_latex
)

echo.
echo [10/10] Compilando reports\academic\main.tex ...
pushd reports\academic
pdflatex -interaction=nonstopmode -halt-on-error main.tex >nul 2>nul
pdflatex -interaction=nonstopmode -halt-on-error main.tex >nul 2>nul
if exist main.pdf (
    echo [10/10]   -^> reports\academic\main.pdf generado
) else (
    echo [10/10]   -^> ERROR: main.pdf no se genero. Revisa reports\academic\main.log
)
popd

echo.
echo [10/10] Compilando reports\dashboard\main.tex ...
pushd reports\dashboard
pdflatex -interaction=nonstopmode -halt-on-error main.tex >nul 2>nul
pdflatex -interaction=nonstopmode -halt-on-error main.tex >nul 2>nul
if exist main.pdf (
    echo [10/10]   -^> reports\dashboard\main.pdf generado
) else (
    echo [10/10]   -^> ERROR: main.pdf no se genero. Revisa reports\dashboard\main.log
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
echo   outputs\predictions\scoreline_predictions.csv
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
