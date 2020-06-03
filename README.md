# An Agent Based Simulation Framework for Epidemic Modeling of Infectious diseases such as COVID-19

The novel COVID-19 virus has disrupted human life across the globe. In a few months, it has spread throughout the world affecting every country on this planet. Efficient monitoring and prediction of the spread can allow the world’s governing bodies to take necessary steps on time to minimize the impact of such diseases. Predicting the spread of infectious diseases is challenging. A conventional approach of modeling such diseases relies on building models on the entire population level, e.g., the SIR and the SEIR models. However, a challenge with the application of these models is the unavailability of the initial parameters such as the rate of spread and recovery. Considering these challenges, in this paper, we take a bottom-up approach and build an agent level simulation framework that models an urban environment along with the mobility of the individuals to predict how the virus spreads for a given geographical region. Our framework provides predictions for the impact of an infectious disease in a given urban environment. On top of that, we can also perform what-if analyses to compare the impact of the disease under different scenarios, e.g., predicting the impact with and without interventions.

Code for the models:
The mobility model: transition_probabilities_simple.csv
The infectiousness model: infectiousnessModel.py
The recovery model: DeathPredictionModel.py
The spatial model parameters can be configured in the config.json file


Parameters:
The config.json file contains the parameters for the simulation.

Dependencies:
Numpy, Pandas


To run, execute: 

python3.6 simulation_without_visualization.py
