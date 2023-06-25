@echo off
cd /d "C:\Packaged-NLP-v1.0"
python -m venv env
call env\Scripts\activate
pip install numpy==1.23.5 pandas==2.0.1 scikit-learn==1.2.2 tensorflow==2.12.0
