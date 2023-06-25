@echo off
cd /d "C:\Packaged-NLP-v1.0"
call env\Scripts\activate
cd information
start cmd /k python dep_version_test.py