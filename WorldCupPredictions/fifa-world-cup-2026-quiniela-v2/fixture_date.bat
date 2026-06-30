@echo off
REM ============================================================
REM  WC2026 - Consulta fechas y partidos (grupos y knockout)
REM ------------------------------------------------------------
REM  1) Por equipo (fase de grupos, igual que antes):
REM     Devuelve la(s) fecha(s) UTC que football_data tiene para
REM     los partidos de un pais/equipo. Esa fecha (UTC) es la que
REM     debes poner en data\raw\manual\wc2026_results.csv para que
REM     el resultado se fusione con el fixture y no cree duplicado.
REM     Solo lista fixtures aun NO jugados (source=football_data).
REM     Ademas muestra los partidos de knockout (si ya estan
REM     definidos) en los que aparece ese equipo.
REM
REM  2) Knockout stage completa:
REM     Lista todo el bracket de data\raw\manual\wc2026_knockout_bracket.csv
REM     con match_id, ronda, fecha y enfrentamiento. Los cupos sin
REM     definir aparecen como '???' y las fechas vacias como
REM     '(sin fecha)'.
REM
REM  Uso:
REM    fixture_date.bat Argentina
REM    fixture_date.bat South Korea
REM    fixture_date.bat korea
REM    fixture_date.bat knockout      (o: ko / bracket / eliminatorias / --knockout)
REM ============================================================

setlocal
cd /d %~dp0

set "WC_ARG=%*"

REM ---- Elegir python del venv si existe ----
set "PY=python"
if exist "..\worldCup\Scripts\python.exe" set "PY=..\worldCup\Scripts\python.exe"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

REM ---- Detectar modo knockout (lista completa del bracket) ----
set "WC_MODE=team"
if /i "%WC_ARG%"=="knockout"      set "WC_MODE=ko"
if /i "%WC_ARG%"=="ko"            set "WC_MODE=ko"
if /i "%WC_ARG%"=="bracket"       set "WC_MODE=ko"
if /i "%WC_ARG%"=="eliminatorias" set "WC_MODE=ko"
if /i "%WC_ARG%"=="-k"            set "WC_MODE=ko"
if /i "%WC_ARG%"=="--knockout"    set "WC_MODE=ko"

if "%WC_MODE%"=="ko" goto :ko

if "%WC_ARG%"=="" (
    echo Uso: fixture_date.bat ^<pais o equipo^>   ^|   fixture_date.bat knockout
    echo   ejemplo: fixture_date.bat Argentina
    echo   ejemplo: fixture_date.bat knockout
    endlocal
    exit /b 1
)

REM ============================================================
REM  Modo equipo: fase de grupos + knockout donde aparezca
REM ============================================================
set "WC_TEAM=%WC_ARG%"

echo.
echo Fixtures de football_data (grupos, no jugados) que coinciden con: %WC_TEAM%
echo ------------------------------------------------------------
%PY% -c "import os,pandas as pd; t=os.environ['WC_TEAM'].strip().strip(chr(34)); df=pd.read_csv('data/interim/matches_unified.csv'); df=df[df['source']=='football_data'].copy(); df['date']=pd.to_datetime(df['date'],errors='coerce'); m=df[df['team_a'].astype(str).str.contains(t,case=False,na=False,regex=False)|df['team_b'].astype(str).str.contains(t,case=False,na=False,regex=False)].dropna(subset=['date']).sort_values('date'); print('\n'.join(str(r.date.date())+' | '+str(r.team_a)+' vs '+str(r.team_b)+'  (grupo '+str(r.group)+', '+str(r.stage)+')' for r in m.itertuples()) if len(m) else 'Sin fixtures de football_data (no jugados) para: '+t)"

echo.
echo Knockout (data\raw\manual\wc2026_knockout_bracket.csv) donde aparece: %WC_TEAM%
echo ------------------------------------------------------------
%PY% -c "import os,pandas as pd; t=os.environ['WC_TEAM'].strip().strip(chr(34)); df=pd.read_csv('data/raw/manual/wc2026_knockout_bracket.csv'); g=lambda v,d:(str(v) if (str(v)!='nan' and str(v).strip()) else d); h=lambda x: t.lower() in str(x).lower(); m=df[df['team_a'].apply(h)|df['team_b'].apply(h)]; print('\n'.join(str(r.match_id).ljust(7)+' | '+str(r.stage).ljust(18)+' | '+g(r.date,'(sin fecha)').ljust(12)+' | '+g(r.team_a,'???')+' vs '+g(r.team_b,'???') for r in m.itertuples()) if len(m) else 'Sin partidos de knockout (definidos) para: '+t)"

echo.
endlocal
exit /b 0

REM ============================================================
REM  Modo knockout: bracket completo
REM ============================================================
:ko
echo.
echo Knockout stage - bracket completo (data\raw\manual\wc2026_knockout_bracket.csv)
echo --------------------------------------------------------------------------
echo  match_id ^| ronda              ^| fecha (UTC)  ^| enfrentamiento
echo --------------------------------------------------------------------------
%PY% -c "import pandas as pd; df=pd.read_csv('data/raw/manual/wc2026_knockout_bracket.csv'); g=lambda v,d:(str(v) if (str(v)!='nan' and str(v).strip()) else d); print('\n'.join(str(r.match_id).ljust(8)+' | '+str(r.stage).ljust(18)+' | '+g(r.date,'(sin fecha)').ljust(12)+' | '+g(r.team_a,'???')+' vs '+g(r.team_b,'???') for r in df.itertuples()))"

echo.
endlocal
exit /b 0
