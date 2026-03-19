@echo off
echo 🚀 Setting up Self-Correcting RAG...
echo.

if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

echo 📥 Installing packages...
call venv\Scripts\activate
pip install -r requirements.txt

echo.
echo ✅ Setup complete!
echo.
echo Run: python crag_demo.py        - Detailed mode
echo      python crag_demo.py --fast - Fast mode
pause