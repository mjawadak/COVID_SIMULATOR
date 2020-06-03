import numpy as np
from scipy import integrate
import pandas as pd
from DeathPredictionModel import DeathPredictionModel

death_prediction_model = DeathPredictionModel()# this code will fetch data from transcend and build an ML model using sklearn. death_prediction_model([age,gender]) -> death_prob

w_shape = 2.83
w_scale = 5.67



def w(x):
    return (w_shape / w_scale) * (x / w_scale)**(w_shape - 1) * np.exp(-(x / w_scale)**w_shape)

def getInfectionProb_daily(days_since_infection):
    if day_since_infection==0:
        return 0
    return integrate.quad(w,days_since_infection-1,days_since_infection)[0]

def getInfectionProb_hourly(days_since_infection_hourly_fraction):
    if days_since_infection_hourly_fraction==0:
        return 0
    return integrate.quad(w,days_since_infection_hourly_fraction-(1/24.0),days_since_infection_hourly_fraction)[0]

def update_agents_after_contact(agents_in_entity_list):
    infected_individuals = []
    df_infected = [] # a data frame that has age and gender of every individual
    for a in agents_in_entity_list:
        if a.is_infected == True:
            infected_individuals.append(a)
            df_infected.append([a.age,a.gender])
    df_infected = pd.DataFrame(df_infected,columns=["age","gender"])


    # Below code is for changing the state from susceptile to infected. For change of state of infected to recovery, rely on the days since infection for every infected individual
    for i in range(len(infected_individuals)):
        #infection_prob = getInfectionProb_daily(infected_individuals[i].days_since_infection)/24.0
        infection_prob = getInfectionProb_hourly(infected_individuals[i].days_since_infection)
        
        for j in range(len(agents_in_entity_list)):
            n_rand = np.random.random()
            if n_rand < infection_prob:
                agents_in_entity_list[j].is_infected = True

    death_probabilities = death_prediction_model.predictDeathProbs(df_infected)


    # Below code is for changing the state from infected to death
    for i in range(len(infected_individuals)):
        r = np.random.random()
        if r < death_probabilities[i]:
            infected_individuals[i].has_died = True


I = []
for i in np.arange(0.,13,1/24.):
    I.append(getInfectionProb_hourly(i))
I = np.array(I)

def getInfectionProb_hourly_cached_np(days_since_infection_hourly_fraction):
    d = days_since_infection_hourly_fraction
    return I[(np.round(d)*24 + (d-np.round(d))*24).astype(int)]



################## TESTING CODE (need to comment it out) ##################

if __name__ == "__main__":
    def get_infected_count():
        counter = 0
        for a in agents_in_entity_list:
            if a.is_infected == True:
                counter += 1
        return counter

    def get_death_count():
        counter = 0
        for a in agents_in_entity_list:
            if a.has_died == True:
                counter += 1
        return counter

    class test_agent():
        def __init__(self):
            self.age = np.random.randint(0,90)
            self.gender = np.random.choice(["female","male"])
            self.is_infected = False
            self.has_died = False
            self.days_since_infection = 0

    agents_in_entity_list = []
    for i in range(50):
        agents_in_entity_list.append(test_agent())
        if i==0:
            agents_in_entity_list[i].is_infected = True
            agents_in_entity_list[i].days_since_infection = 5
        #if i==2:
            #agents_in_entity_list[i].is_infected = True
            #agents_in_entity_list[i].days_since_infection = 3

    count_infected = get_infected_count()
    count_death = get_death_count()
    print("before:"," infected_count = ",count_infected,", death_count = ",count_death)
    
    for i in range(12*24): # test the code for 6 hours
        update_agents_after_contact(agents_in_entity_list)
    
    count_infected = get_infected_count()
    count_death = get_death_count()
    print("after "+str(i)+" hours:"," infected_count = ",count_infected,", death_count = ",count_death)

