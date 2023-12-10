install:
	pip install -r requirements.txt 
    
plotly-extension:
	jupyter labextension install jupyterlab-plotly@4.9.0

gitignore:
	touch .gitignore && echo ".gitignore" >> .gitignore && echo "*__pycache__/" >> .gitignore && echo "*ipynb_checkpoints/" >> .gitignore 
    
setup: install plotly-extension gitignore

nbstripout:
	nbstripout *.ipynb

black:
	black *.py
    
clean:
	rm -r .ipynb_checkpoints/
