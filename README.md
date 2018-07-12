# ccnss_miniproject_3



conda env create --file environment.yml

conda install --file requirements.txt -y

pip install -e .

pytest --cov-report=html --cov=relaxrender --ignore=tests/test_relaxrender.py tests
