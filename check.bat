@echo off
REM Pre-flight checks -- run this before copying files to the BlueIris box.
REM Double-click it, or run `check.bat` from a terminal. Pauses at the end so
REM you can read the result when launched by double-click.

cd /d "%~dp0"

echo == Syntax check ==
for %%f in (*.py) do (
  python -m py_compile "%%f"
  if errorlevel 1 goto fail
)

echo.
echo == Tests ==
python -m pytest -q
if errorlevel 1 goto fail

echo.
echo All checks passed -- safe to deploy.
pause
exit /b 0

:fail
echo.
echo *** CHECKS FAILED -- fix the above before deploying. ***
pause
exit /b 1
