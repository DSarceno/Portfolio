@echo off
REM ============================================================
REM  WC2026 - Actualizacion rapida de jornada
REM ------------------------------------------------------------
REM  Ingiere resultados recientes y regenera ratings, features,
REM  modelos, predicciones, marcadores, quinielas y simulacion.
REM
REM  Uso:
REM    update_matchday.bat                  -> usa data\raw\manual\wc2026_results.csv
REM    update_matchday.bat ruta\a\res.csv   -> usa ese CSV de resultados
REM    update_matchday.bat 2026-06-15       -> baja resultados de esa fecha (football-data.org)
REM
REM  NO reconstruye snapshots de valor de plantilla: son estaticos
REM  durante el torneo. Si una salida da PermissionError, cierra
REM  ese CSV en Excel/editor y vuelve a correr.
REM ============================================================

setlocal enabledelayedexpansion
cd /d %~dp0

set "N_RUNS=10000"
set "ARG=%~1"
if "%ARG%"=="" set "ARG=data\raw\manual\wc2026_results.csv"

echo.
echo ============================================================
echo  WC2026 - ACTUALIZACION DE JORNADA
echo  Inicio: %DATE% %TIME%
echo ============================================================

REM ---- Activar venv ----
if exist "..\worldCup\Scripts\activate.bat" goto :act_wc
if exist ".venv\Scripts\activate.bat" goto :act_venv
echo [setup] Sin venv detectado; usando python del sistema
goto :venv_done
:act_wc
call "..\worldCup\Scripts\activate.bat"
echo [setup] venv ..\worldCup activado
goto :venv_done
:act_venv
call ".venv\Scripts\activate.bat"
echo [setup] venv .venv activado
:venv_done

python --version >nul 2>nul
if errorlevel 1 (
    echo ERROR: python no encontrado en PATH
    goto :failure
)

REM ---- Paso 1: ingerir resultados ----
echo.
echo  [1/9] Ingiriendo resultados
if exist "%ARG%" goto :use_csv
echo  [1/9]   -^> modo fecha: %ARG%
python scripts\update_after_matchday.py --date %ARG%
if errorlevel 1 goto :failure_update
goto :after_ingest
:use_csv
echo  [1/9]   -^> CSV: %ARG%
python scripts\update_after_matchday.py --manual-csv "%ARG%"
if errorlevel 1 goto :failure_update
:after_ingest

REM ---- Paso 2: ingerir bracket de knockout ----
REM  Mete los cruces de cada ronda (R16, 4tos, ...) que ya hayas llenado en
REM  data\raw\manual\wc2026_knockout_bracket.csv. Es seguro re-correrlo: NO
REM  pisa los partidos que ya tienen resultado (esos se respetan).
echo.
echo  [2/9] Ingiriendo bracket de knockout
python scripts\ingest_knockout_bracket.py
if errorlevel 1 goto :failure_step

REM ---- Paso 3: ratings ----
REM  Elo/pi/form SI se actualizan con cada resultado (son baratos y estables).
echo.
echo  [3/9] Reconstruyendo ratings
python scripts\build_ratings.py
if errorlevel 1 goto :failure_step

REM  NOTA: los modelos (XGBoost/Poisson/multinomial/calibracion) NO se reentrenan
REM  durante el torneo. Se entrenaron y validaron antes del Mundial y se congelan:
REM  reentrenar con un puñado de partidos nuevos los desestabiliza (paso a paso el
REM  multinomial llego a saturar a ~99% al equipo visitante). Para reentrenar a
REM  proposito: corre  scripts\run_pipeline.py  y  scripts\train_models.py  a mano.

REM ---- Paso 4: predicciones H/D/A (grupos) ----
echo.
echo  [4/9] Prediciendo H/D/A (grupos, si quedan)
python scripts\predict_group_stage.py
if errorlevel 1 goto :failure_step

REM ---- Paso 5: predicciones knockout ----
echo.
echo  [5/9] Prediciendo eliminatorias
python scripts\predict_knockout.py
if errorlevel 1 goto :failure_step

REM ---- Paso 6: marcadores ----
echo.
echo  [6/9] Prediciendo marcadores mas probables
python scripts\predict_scorelines.py
if errorlevel 1 goto :failure_step

REM ---- Paso 7: quinielas ----
echo.
echo  [7/9] Exportando quinielas
python scripts\export_quiniela_sheet.py
if errorlevel 1 goto :failure_step

REM ---- Paso 8: simulacion ----
echo.
echo  [8/9] Simulando torneo Monte-Carlo (n_runs=%N_RUNS%)
python scripts\simulate_tournament.py --n-runs %N_RUNS%
if errorlevel 1 goto :failure_step

REM ---- Paso 9: snapshot datado de outputs ----
REM  Archiva una copia datada de outputs/ en outputs\snapshots\<fecha>\ con un
REM  manifest.json (commit, partidos jugados, top-5 al titulo). Los outputs en
REM  vivo se sobrescriben en cada corrida; el snapshot preserva el pronostico de
REM  esta jornada y alimenta la comparacion pre-Mundial vs lo que paso. Re-correr
REM  el mismo dia refresca el snapshot del dia. Que falle NO aborta la jornada.
echo.
echo  [9/9] Guardando snapshot datado de outputs
python scripts\snapshot_outputs.py
if errorlevel 1 echo  [9/9]   AVISO: el snapshot fallo (no critico). Revisa logs\updates\.

echo.
echo ============================================================
echo  ACTUALIZACION COMPLETADA
echo  Fin: %TIME%
echo ============================================================
echo Revisa:
echo   outputs\predictions\group_stage_predictions.csv
echo   outputs\predictions\scoreline_predictions.csv
echo   outputs\picks\quiniela_balanced.csv
echo   outputs\simulations\championship_probabilities.csv
echo   outputs\snapshots\^<fecha^>\  (copia datada de esta jornada)
endlocal
exit /b 0

:failure_update
echo.
echo ERROR: fallo la ingesta de resultados. Revisa logs\updates\.
endlocal
exit /b 1

:failure_step
echo.
echo ERROR: fallo un paso de regeneracion ^(posible PermissionError: cierra el CSV en Excel^). Revisa logs\.
endlocal
exit /b 1

:failure
echo.
echo ERROR en el setup.
endlocal
exit /b 1
