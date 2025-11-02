# query-analysis-plot-agent
LLM agent to query analyze and plot results

# Install requirements
pip install -r requirements.txt
or 
pip install --no-cache-dir --force-reinstall -r requirements.txt

# Run
python cli.py "Show top 5 artists by sales" --database chinook

# Test

pytest -s -v tests/test_functional_chinook.py

## RUn backend and front end 

python app.py

Open index.html 