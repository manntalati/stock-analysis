\
@echo off
setlocal
set PROJ=%~dp0..
cd /d %PROJ%
call .\.venv\Scripts\activate
python -m stocklab.cli build-universe --source sp500
python -m stocklab.cli fetch --start 2015-01-01
python -m stocklab.cli features
python -m stocklab.cli predict
python -m stocklab.cli score
python -m stocklab.cli backtest --start 2018-01-01 --end 2024-12-31
python -m stocklab.cli report-html
for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set today=%%c-%%a-%%b
python -m stocklab.cli picks --date %today% --top 20
endlocal
