@echo off
cd /d "C:\Packaged-NLP-v1.0"
call env\Scripts\activate
start cmd /k python master.py