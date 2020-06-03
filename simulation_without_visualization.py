import infectiousnessModel
from infectiousnessModel import getInfectionProb_daily,getInfectionProb_hourly,getInfectionProb_hourly_cached_np
from location_probability_matrix import getNextLocType
from DeathPredictionModel import DeathPredictionModel
import numpy as np
import pandas as pd
import datetime
import time
import matplotlib.pyplot as plt
import sys
import json



def get_infected_count():
    counter = 0
    for a in agent_list:
        if a.is_infected == True:
            counter += 1
    return counter


def get_death_count():
    counter = 0
    for a in agent_list:
        if a.has_died == True:
            counter += 1
    return counter


class Agent():
    def __init__(self,id):
        self.id = id
        self.age = np.random.randint(5, 90)
        self.gender = np.random.randint(0,2)#np.random.choice(["female", "male"]) # 0->female,1->male
        self.is_infected = False
        self.has_died = False
        self.has_recovered = False
        self.days_since_infection = 0
        self.currentLocationID = -1
        self.currentLocationType = "residential"
        self.infection_date = 0

    def update_infection_state(self):
        if self.is_infected == True:
            if self.days_since_infection > 12:
                self.has_recovered = True
                self.is_infected = False
                infected_list.remove(self.id)
                recovered_list.append(self.id)
            else:
                self.days_since_infection += 1.0/24.0


class Entity():
    def __init__(self,cat):
        self.category = cat
        self.x = 0
        self.y = 0
        self.current_list_of_agent_ids = []


def get_random_entity_index(loc_type):
    if loc_type == 'residential':
        return np.random.randint(0,NUM_RESIDENCES)
    if loc_type == 'school':
        return np.random.randint(NUM_RESIDENCES, NUM_RESIDENCES+NUM_SCHOOLS)
    if loc_type == 'retail':
        return np.random.randint(NUM_RESIDENCES+NUM_SCHOOLS, NUM_RESIDENCES+NUM_SCHOOLS+NUM_RETAILS)
    if loc_type == 'employment':
        return np.random.randint(NUM_RESIDENCES+NUM_SCHOOLS+NUM_RETAILS, NUM_RESIDENCES+NUM_SCHOOLS+NUM_RETAILS+NUM_WORKPLACES)


def update_locations(current_datehour_in_simulation):
    #for e in entity_list:
        #e.current_list_of_agent_ids = []

    for i in range(len(agent_list)):
        t1 = time.time()

        next_loc_type = getNextLocType(agent_list[i].age, current_datehour_in_simulation,intervention_flag,LOCKED_ENTITIES)
        t2 = time.time()
        if (next_loc_type != agent_list[i].currentLocationType or agent_list[i].currentLocationID==-1 ) and agent_list[i].has_died==False:
            #print(i,agent_list[i].currentLocationID,entity_list[agent_list[i].currentLocationID].current_list_of_agent_ids.__len__())

            if agent_list[i].currentLocationID!=-1:
                entity_list[agent_list[i].currentLocationID].current_list_of_agent_ids.remove(i)

            agent_list[i].currentLocationID = get_random_entity_index(next_loc_type)
            t3 = time.time()
            agent_list[i].currentLocationType = next_loc_type
            t4 = time.time()
            #print(i,next_loc_type,agent_list[i].currentLocationID,len(entity_list))
            entity_list[agent_list[i].currentLocationID].current_list_of_agent_ids.append(i)
            t5 = time.time()
            #print(t2-t1,t3-t2,t4-t3,t5-t4)

def update_infected_days_count(datehour):
    for agent in agent_list:
        agent.update_infection_state(datehour)

def update_agents_after_contact(agents_in_entity_list):
    if len(agents_in_entity_list)>0:
        t1 = time.time()
        infected_individuals = []
        df_infected = []  # a data frame that has age and gender of every individual
        for a in agents_in_entity_list:
            if agent_list[a].is_infected == True:
                infected_individuals.append(a)
                df_infected.append([agent_list[a].age, agent_list[a].gender])
        df_infected = np.array(df_infected)
        #df_infected = pd.DataFrame(df_infected, columns=["age", "gender"])
        #print(df_infected.shape)
        # Below code is for changing the state from susceptile to infected. For change of state of infected to recovery, rely on the days since infection for every infected individual

        t2 = time.time()
        infection_prob = 0
        for i in range(len(infected_individuals)):
            # infection_prob = getInfectionProb_daily(infected_individuals[i].days_since_infection)/24.0
            infection_prob += getInfectionProb_hourly_cached_np(agent_list[infected_individuals[i]].days_since_infection)

        for j in range(len(agents_in_entity_list)):
            if agent_list[agents_in_entity_list[j]].is_infected == False and \
                    agent_list[agents_in_entity_list[j]].has_recovered == False and \
                    agent_list[agents_in_entity_list[j]].has_died == False:
                n_rand = np.random.random()
                if n_rand < infection_prob:
                        agent_list[agents_in_entity_list[j]].is_infected = True
                        infected_list.append(agents_in_entity_list[j])

        t3 = time.time()
        if df_infected.shape[0]>0:
            #print(df_infected)
            t4 = time.time()
            death_probabilities = death_prediction_model.predictDeathProbs(df_infected)
            t5 = time.time()
            # Below code is for changing the state from infected to death

            for i in range(len(infected_individuals)):
                r = np.random.random()
                if r < death_probabilities[i]:
                    if agent_list[infected_individuals[i]].has_died == False:
                        agent_list[infected_individuals[i]].has_died = True
                        agent_list[infected_individuals[i]].days_since_infection = 0
                        agent_list[infected_individuals[i]].is_infected = False
                        agent_list[infected_individuals[i]].has_recovered = False
                        death_list.append(infected_individuals[i])
                        infected_list.remove(infected_individuals[i])
            t6 = time.time()
            #print(t3-t2,t2-t1,len(infected_individuals))

        t7 = time.time()
        #if t7-t1 > 0.5:
        #print("-",len(agents_in_entity_list),len(infected_individuals),t7-t3,t3-t1)





'''
SCALE_FACTOR = 1000.0
NUM_SCHOOLS = int(3000 / SCALE_FACTOR)
NUM_RESIDENCES = int(3560000 / SCALE_FACTOR)
NUM_RETAILS = int(43000 / SCALE_FACTOR)
NUM_WORKPLACES = int(1100000 / SCALE_FACTOR)
POPULATION = int(8910000 / SCALE_FACTOR)
INTERVENTION_INFO = {"flag":False,"locked_entities":[],"counter":0,"threshold_infected":1000}'''

with open("config.json","r") as config_file:
    config = json.load(config_file)["parameters"]
    SCALE_FACTOR = config["SCALE_FACTOR"]
    NUM_SCHOOLS = int(config["NUM_SCHOOLS"]/float(SCALE_FACTOR))
    NUM_RESIDENCES = int(config["NUM_RESIDENCES"]/float(SCALE_FACTOR))
    NUM_RETAILS = int(config["NUM_RETAILS"]/float(SCALE_FACTOR))
    NUM_WORKPLACES = int(config["NUM_WORKPLACES"]/float(SCALE_FACTOR))
    POPULATION = int(config["POPULATION"]/float(SCALE_FACTOR))
    INTERVENTION_INFO = config["INTERVENTION_INFO"]
    LOCKED_ENTITIES = INTERVENTION_INFO["locked_entities"]


intervention_flag = False
helper_counter = 0 # to change the above flag only once
# create entity list
entity_list = []
for i in range(NUM_RESIDENCES):
    entity_list.append(Entity("residential"))
for i in range(NUM_SCHOOLS):
    entity_list.append(Entity("school"))
for i in range(NUM_RETAILS):
    entity_list.append(Entity("retail"))
for i in range(NUM_WORKPLACES):
    entity_list.append(Entity("employment"))

# create agent list
agent_list = []
infected_list = []
recovered_list = []
death_list = []
for i in range(POPULATION):
    agent_list.append(Agent(i))
    if i == 0:
        agent_list[i].age = 20 # to ensure that the person does not die
        agent_list[i].is_infected = True
        agent_list[i].days_since_infection = 0
        infected_list.append(i)





death_prediction_model = DeathPredictionModel()# this code will fetch data from transcend and build an ML model using sklearn. death_prediction_model([age,gender]) -> death_prob
'''
# TESTING CODE
I = []
for i in np.arange(0.,13,1/24.):
    I.append(getInfectionProb_hourly(i))
t1 = time.time()
I = np.array(I)
#death_prediction_model.predictDeathProbs(np.ones((1000000,2)))

d=np.array([1,1,1,1,1,1,1,1,1,1,1])
#I[int(d)*24+int(round((d - int(d))*24))]
#Z = I[(np.round(d)*24 + (d-np.round(d))*24).astype(int)]
Z= []
for dd in d:
    Z.append(getInfectionProb_hourly(10))
t2 = time.time()
print(1000000*(t2-t1))
print(Z)

print(getInfectionProb_hourly_cached_np([10,1]))
print(getInfectionProb_hourly(10))

sys.exit()'''


count_infected = get_infected_count()
count_death = get_death_count()
print("before:", " infected_count = ", count_infected, ", death_count = ", count_death)
current_date = datetime.datetime(year=2020,month=1,day=1,hour=0)
total_time = 0
OUTPUT = []
cumulative_infected_count = 0
intervention_day = 0
for t in range(48 * 24):  # test the code

    t1 = time.time()
    current_date = current_date + datetime.timedelta(hours=1)
    update_locations(current_date)
    t2 = time.time()
    #print(current_date,agent_list[0].currentLocationType,agent_list[0].currentLocationID)
    for e in entity_list:
        update_agents_after_contact(e.current_list_of_agent_ids)
    t3 = time.time()

    for i in infected_list:
        agent_list[i].update_infection_state()
    t4 = time.time()

    total_time += t4-t1

    OUTPUT.append([current_date.strftime("%Y-%m-%d"),current_date.strftime("%H:%M:%S"),infected_list.__len__(),recovered_list.__len__(),death_list.__len__(),t4-t1])
    print(current_date,infected_list.__len__(),recovered_list.__len__(),death_list.__len__(),t4-t1,t3-t2)

    if helper_counter == 0 and infected_list.__len__() > INTERVENTION_INFO["threshold_infected"]:
        intervention_flag = True
        intervention_day = current_date
        helper_counter = 1
    '''if infected_list.__len__() > INTERVENTION_INFO["threshold_infected"] and INTERVENTION_INFO["counter"] == 0:
        INTERVENTION_INFO["flag"] = True
        INTERVENTION_INFO["locked_entities"] = ["school","employment","retail"]
        INTERVENTION_INFO["counter"] += 1
        intervention_day = current_date'''

    '''if infected_list.__len__() < 2000 and INTERVENTION_INFO["counter"] = 1:
        INTERVENTION_INFO["flag"] = False
        INTERVENTION_INFO["locked_entities"] = []
        INTERVENTION_INFO["counter"] += 1'''

    #if t4-t1>0.6:
        #print("--",t2-t1,t3-t2,t4-t3)

    #print(infected_list.__len__(),infected_list[0],entity_list[agent_list[infected_list[0]].currentLocationID].current_list_of_agent_ids)

OUTPUT = pd.DataFrame(OUTPUT,columns = ["Date","Time","Infected_count","Recovered_count","Death_count","time_taken_secs"])
file_name = "OUTPUT_"+str(int(time.time())) +"_"+str(INTERVENTION_INFO["flag"])
OUTPUT.to_csv("simulation_output/"+file_name+".csv")
count_infected = get_infected_count()
count_death = get_death_count()
#print("after " + str(i) + " hours:", " infected_count = ", count_infected, ", death_count = ", count_death, ",total_time =", total_time)

infected_count = recovered_list.__len__() + death_list.__len__()
recovered_count = recovered_list.__len__()
death_count = death_list.__len__()
output_string="""
-------------------------------- REPORT ---------------------------------
Intervention = {}
Output file name = {}
Number of simulation hours run = {} ({} days)
Total run time in seconds = {} ({} average secs taken to simulate one hour)
Total Population = {}
Number of residences = {}
Number of schools = {}
Number of workplaces = {}
Number of retail shops = {}
SCALE_FACTOR = {}
Number of Infected = {} ({}% of total population)
Number of Recovered = {} ({}% of total population, {}% of total infected)
Number of Deaths = {} ({}% of total population, {}% of total infected)
-------------------------------------------------------------------------
""".format(str(INTERVENTION_INFO),
           file_name,
           str(t),
           str(int(t/24)),
           str(int(total_time)),
           "{:.2f}".format(total_time/t),
           str(POPULATION),
           str(NUM_RESIDENCES),
           str(NUM_SCHOOLS),
           str(NUM_WORKPLACES),
           str(NUM_RETAILS),
           str(SCALE_FACTOR),
           str(infected_count),
           str(round(100*infected_count/ float(POPULATION))),
           str(recovered_count),
           str(round(100*recovered_count / float(POPULATION))),
           str(round(100*recovered_count / float(infected_count))),
           str(death_count),
           str(round(100*death_count / float(POPULATION))),
           str(round(100*death_count / float(infected_count)))
           )
with open("simulation_output/"+file_name+".txt","w") as outfile:
    outfile.write(output_string)

print(output_string)

OUTPUTgrouped = OUTPUT.groupby("Date").mean()

plt_title = "Without Intervention"
if INTERVENTION_INFO["flag"] == True:
    plt_title = "With Intervention ("+", ".join(INTERVENTION_INFO["locked_entities"])+" locked on "+str(intervention_day.date())+")"

plt.figure(figsize=[9,4])
plt.title(plt_title)
plt.plot(OUTPUTgrouped.index,100*OUTPUTgrouped["Infected_count"]/POPULATION,label="Active cases")
plt.plot(OUTPUTgrouped.index,100*OUTPUTgrouped["Recovered_count"]/POPULATION,label="Recovered")
plt.plot(OUTPUTgrouped.index,100*OUTPUTgrouped["Death_count"]/POPULATION,label="Death")
plt.xticks(rotation=90)
plt.ylim([0,100])
plt.ylabel("% of Total Population")
plt.legend()
plt.tight_layout()
plt.savefig("simulation_output/"+file_name+".png")