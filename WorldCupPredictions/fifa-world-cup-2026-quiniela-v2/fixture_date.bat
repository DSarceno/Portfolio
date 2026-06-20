@echo off
REM ============================================================
REM  WC2026 - Consulta la fecha UTC de un fixture en football_data
REM ------------------------------------------------------------
REM  Devuelve la(s) fecha(s) que football_data tiene para los
REM  partidos de un pais/equipo. Esa fecha (UTC) es la que debes
REM  poner en data\raw\manual\wc2026_results.csv para que el
REM  resultado se fusione con el fixture y no cree un duplicado.
REM
REM  Solo lista fixtures aun NO jugados (source=football_data);
REM  los ya ingresados desaparecen de la lista porque se fusionan.
REM
REM  Uso:
REM    fixture_date.bat Argentina
REM    fixture_date.bat South Korea
REM    fixture_date.bat korea
REM ============================================================

setlocal
cd /d %~dp0

set "WC_TEAM=%*"
if "%WC_TEAM%"=="" (
    echo Uso: fixture_date.bat ^<pais o equipo^>
    echo   ejemplo: fixture_date.bat Argentina
    endlocal
    exit /b 1
)

REM ---- Elegir python del venv si existe ----
set "PY=python"
if exist "..\worldCup\Scripts\python.exe" set "PY=..\worldCup\Scripts\python.exe"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

echo.
echo Fixtures de football_data que coinciden con: %WC_TEAM%
echo ------------------------------------------------------------
%PY% -c "import os,pandas as pd; t=os.environ['WC_TEAM'].strip().strip(chr(34)); df=pd.read_csv('data/interim/matches_unified.csv'); df=df[df['source']=='football_data'].copy(); df['date']=pd.to_datetime(df['date'],errors='coerce'); m=df[df['team_a'].astype(str).str.contains(t,case=False,na=False,regex=False)|df['team_b'].astype(str).str.contains(t,case=False,na=False,regex=False)].dropna(subset=['date']).sort_values('date'); print('\n'.join(str(r.date.date())+' | '+str(r.team_a)+' vs '+str(r.team_b)+'  (grupo '+str(r.group)+', '+str(r.stage)+')' for r in m.itertuples()) if len(m) else 'Sin fixtures de football_data (no jugados) para: '+t)"

echo.
endlocal
exit /b 0
