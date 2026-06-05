# Streamlit via venv

- python -m venv myvenv
- source myvenv/bin/activate
- pip install -r requirements.txt
- pip list
- streamlit run app.py (stop streamlit by Ctrl + C)
- deactivate
- exit

to use the myvenv for info.ipynb
- pip install ipykernel
Register the env as notebook kernel
- python -m ipykernel install --user --name=myvenv --display-name "Python (myvenv)"
Create Python environment | Enter Interpreter path
- /workspaces/Resume-projects/Streamlit via venv/myvenv/bin/