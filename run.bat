python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python test_lossless_alg.py
