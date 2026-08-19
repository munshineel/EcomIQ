# Launch the EcomIQ dashboard
$PY = "C:\DS-AI-Spiced\Product_Review-Sentiment_Analysis\.venv\Scripts\python.exe"
Set-Location $PSScriptRoot
& $PY -m streamlit run app\Home.py
