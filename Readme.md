# How to run 
- Activate the venv by running `source .venv/bin/activate`
- Install the requirement file with `pip install requirements.txt`
- To start the api run `python3.10 main.py` by default the app will start on port 8005 , you can change that in  `main.py`
- To have a visualization and test all the endpoints and the API routes available =>  http://localhost:8005/docs

# Model info :

- Dataset : the model is built based on the csv file `Wines.csv`
- Modelling : we use the model of Random Forest because it's a multi class problem
- Biggest weight in the model : alcohol / sulphates / volatile acidity
- The model has an accuracy of 74%


# Made BY:
- AIT TMILLA yassir 
- ROSAY Cillian
