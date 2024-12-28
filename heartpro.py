import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


df=pd.read_csv("hearts.csv")
print(df)

#labelEncoder-- into numerical
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


df['Gender']=le.fit_transform(df['Gender'])
df['chestPainType']=le.fit_transform(df['ChestPainType'])
df['RestingECG']=le.fit_transform(df['RestingECG'])
df['ExerciseAngina']=le.fit_transform(df['ExerciseAngina'])
df['ST_Slope']=le.fit_transform(df['ST_Slope'])
print(df)


#create input and output variables

x = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']
print(x)
print(y)

#raining models
#testing and raining
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=12)


from sklearn.naive_bayes import GaussianNB
NB=GaussianNB()
NB.fit(x_train,y_train)
y_pred = NB.predict(x_test)        #MODEL EVALUATION
print(y_pred)
print(y_test)

#NOW WE SHOULD COMPARE TO FIND ACCURACY
from sklearn.metrics import accuracy_score
print("ACCURACY IS....:",accuracy_score(y_test,y_pred))

import pickle    #save the model
pickle.dump(NB,open('model.pkl','wb'))


testPredicition=NB.predict[[23,1,1,23,43,23,2,2,54,5,43]]
if testPredicition ==1:
    print("the patient has heart disease")
else:
    print("the patient is normal")



