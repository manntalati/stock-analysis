# Dashboard + Automation Pack

## Install
. .venv/bin/activate
pip install -r requirements.txt
pip install streamlit plotly

## Run Dashboard
streamlit run app/streamlit_app.py

## Daily Automation (cron on macOS/Linux)
chmod +x bin/daily.sh
crontab -e
# add line (edit paths):
# 15 17 * * MON-FRI /bin/bash -lc 'cd /ABS/PATH/TO/PROJECT && . .venv/bin/activate && bin/daily.sh >> logs/daily.log 2>&1'

## Windows Task Scheduler
Use `bin/daily.bat` as the action. Set "Start in" to the project folder.

## GitHub Actions (optional)
Push `.github/workflows/daily.yml` to your repo and configure Actions secrets if needed.
