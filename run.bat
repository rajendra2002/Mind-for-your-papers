@echo off
echo Starting Multi-Modal RAG App...
python -m pip install -r requirements.txt
python -m streamlit run src/app.py
pause
