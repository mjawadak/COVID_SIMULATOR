import teradatasql
import getpass
import numpy as np
import pandas as pd

##### death prediction model
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

class DeathPredictionModel():

	def __init__(self):
		##### death prediction model
		'''username = 'mk250133'
		password = getpass.getpass(prompt= 'Password:', stream = None)
		con = teradatasql.connect(host = "tdprd2.td.teradata.com", user = username, password = password,logmech='LDAP') 
		del password
		cur = con.cursor()
		cur.execute("""select age,gender,status from ADLDEMO_COVID19_GDC.cases_consolidated 
			where (status = 'Deceased' or status = 'Recovered') and age is not null and gender is not null;""")
		res = cur.fetchall()
		df = pd.DataFrame(res,columns=["age","gender","status"])'''

		df = pd.read_csv("transcendData/deathPredictionModelData.csv")
		df["gender"]= df["gender"].str.lower()
		df["status"]= df["status"].str.lower()
		self.le_gender = preprocessing.LabelEncoder()
		self.le_gender.fit(df["gender"].unique())
		self.le_death = preprocessing.LabelEncoder()
		self.le_death.fit(df["status"].unique())

		df["gender_int"]= self.le_gender.transform(df["gender"])
		df["status_int"]= self.le_death.transform(df["status"])

		# If you want to use grid search, you can use the below code:
		'''rf = RandomForestClassifier()

		parameters = {'min_samples_leaf':(1,5,10,15,20), 
		              'max_depth':[5,10,15],
		              'criterion':['gini','entropy'],
		              'n_estimators':[10,50,100,200]}
		clf_gs = GridSearchCV(rf, parameters)
		clf_gs.fit(df[["age","gender_int"]],df["status_int"])

		self.clf = clf_gs.best_estimator_'''

		#self.clf = RandomForestClassifier(min_samples_leaf=5,max_depth=10)
		self.clf = DecisionTreeClassifier(min_samples_leaf=25, max_depth=3)

		self.clf.fit(df[["age","gender_int"]].values,df["status_int"].values)
		#cur.close()




	def predictDeathProbs(self,df):

		#inputs = np.vstack((df["age"].values,self.le_gender.transform(df["gender"]))).T
		#print("--",df)
		inputs = df
		death_probabilities = self.clf.predict_proba(inputs)[:,0] # a list of death probabilites for each infected individual
		# Below, dividing each prob with the average total number of infected days (12) and then by 24 hours. This is because this function is called every hour
		return death_probabilities/12.0/24.0

################## TESTING CODE (need to comment it out) ##################

if __name__ == "__main__":
	death_prediction_model = DeathPredictionModel()
	print(death_prediction_model.predictDeathProbs([[0,1],
													[10, 1],
													[20, 1],
													[30, 1],
													[40, 1],
													[50, 1],
													[60, 1],
													[70, 1],
													[80, 1],
													[90, 1],
													]))