@echo off
ECHO Killing any process that is using port 8080...

:: This command finds any process using port 8080 and extracts its PID (the 5th token)
:: It then runs taskkill on that PID. The pipe symbol | must be escaped with ^
FOR /F "tokens=5" %%a IN ('netstat -ano ^| findstr :8080') DO (
    ECHO Found process with PID %%a. Terminating...
    taskkill /F /PID %%a
)

ECHO Port 8080 has been cleared.
ECHO.
ECHO Starting the Python server...
python app.py