@echo off
setlocal EnableDelayedExpansion

set "OUTDIR=gptfiles"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

for /r "%CD%" %%F in (*.js *.css *.py *.html) do (
  set "dir=%%~dpF"
  echo !dir! | findstr /i /c:"\%OUTDIR%\" >nul
  if errorlevel 1 (
    set "full=%%~fF"
    set "rel=!full:%CD%\=!"
    set "name=!rel:\=_!"
    if "!name:~0,1!"=="_" set "name=!name:~1!"
    set "destname=!name!.txt"
    copy /y "%%~fF" "%OUTDIR%\!destname!" >nul
  )
)

echo Done. Files copied to "%OUTDIR%".
endlocal

