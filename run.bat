@echo off
echo Installing required packages...
pip install opencv-python numpy scikit-learn matplotlib gradio tabulate -q

echo.
echo Starting the app...
python app.py
pause
