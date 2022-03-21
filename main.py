import pandas as pd
import streamlit as st
import joblib
import pickle
import datetime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import  StandardScaler

st.sidebar.image(r".\images.jpg")
st.title("House price prediction")
tab=st.sidebar.columns(2)
today=datetime.date.today()
tab[0].date_input("today",today)
tab[1].time_input("time")

room_number=st.number_input("number of rooms",min_value=1,max_value=10)
bathroom_number=st.number_input("number of bathroom",min_value=1,max_value=8)
category=st.selectbox("choose the category",['Appartements', 'Locations de vacances',
       'Magasins, Commerces et Locaux industriels', 'Maisons et Villas',
       'Colocations', 'Bureaux et Plateaux'])
city=st.selectbox("choose the city",['Ariana', 'Béja', 'Ben arous', 'Bizerte', 'Gabès', 'Gafsa',
       'Jendouba', 'Kairouan', 'Kasserine', 'Kébili', 'La manouba',
       'Le kef', 'Mahdia', 'Médenine', 'Monastir', 'Sidi bouzid',
       'Siliana', 'Sousse', 'Tataouine', 'Tozeur', 'Zaghouan', 'Sfax',
       'Nabeul', 'Tunis'])
types=st.selectbox("choose the type",['À Vendre', 'À Louer'])
surface=st.number_input("choose the size",min_value=30 ,max_value=1000)

c=st.columns(10)
pred=c[5].button("predict")

#Separate categorical values and Numerical Values
Cat_Col = ['category','city','type']
Num_Col = ['room_count','bathroom_count' , 'size']

Pipeline = ColumnTransformer([
    ("num", StandardScaler(), Num_Col),
    ('cat', OrdinalEncoder(),Cat_Col)
])

df = pd.read_csv("Immobiliers.csv")

Pipeline.fit_transform((df.drop(['price','log_price','region'],axis=1)))

with open("pipe.h5","rb") as f:
    pipe=pickle.load(f)
    dftest=pd.DataFrame([[category,float(room_number),int(bathroom_number),float(surface),types,city]],columns=['category', 'room_count', 'bathroom_count', 'size', 'type',
       'city'])
    #st.dataframe(dftest)
    v=Pipeline.transform(dftest)
model=joblib.load("best.h5")
prediction=model.predict(v)
if pred :
    st.columns(10)
    st.balloons()
    c[5].success("{:.2f}".format(prediction[0]))
    print(prediction)