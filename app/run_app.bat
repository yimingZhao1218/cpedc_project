@echo off
echo ================================
echo CPEDC PINN 可视化平台启动中...
echo ================================
echo.

REM 激活虚拟环境（如果存在）
if exist "..\\.venv\\Scripts\\activate.bat" (
    echo 激活虚拟环境...
    call "..\\.venv\\Scripts\\activate.bat"
)

echo 启动 Streamlit 应用...
streamlit run streamlit_app.py

pause
