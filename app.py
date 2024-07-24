import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Cargar el dataset original para obtener las opciones categóricas
data = pd.read_csv('data_evaluacion.csv')
data.columns = data.columns.str.strip()  # Limpiar los nombres de las columnas

# Renombrar columnas
data.columns = [
    'Edad', 'Tipo_Empleo', 'Ingresos', 'Educacion', 'Anios_Educacion',
    'Estado_Civil', 'Ocupacion', 'Relacion', 'Raza', 'Genero',
    'Ganancia_Capital', 'Perdida_Capital', 'Horas_Por_Semana', 'Pais', 'Ingreso_Anual'
]

# Obtener opciones únicas para variables categóricas
workclass_options = data['Tipo_Empleo'].unique().tolist()
education_options = data['Educacion'].unique().tolist()
marital_status_options = data['Estado_Civil'].unique().tolist()
occupation_options = data['Ocupacion'].unique().tolist()
relationship_options = data['Relacion'].unique().tolist()
race_options = data['Raza'].unique().tolist()
sex_options = data['Genero'].unique().tolist()
native_country_options = data['Pais'].unique().tolist()

# Cargar el modelo
pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

# Crear escalador
scaler = StandardScaler()

# Crear label encoders y ajustarlos a los datos
label_encoders = {}
for column in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Función para hacer predicciones
def predict_income(age, wgfnlt, education_num, capital_gain, capital_loss, hours_per_week, workclass, education, marital_status, occupation, relationship, race, sex, native_country):
    # Transformar las variables categóricas
    workclass = label_encoders['Tipo_Empleo'].transform([workclass])[0]
    education = label_encoders['Educacion'].transform([education])[0]
    marital_status = label_encoders['Estado_Civil'].transform([marital_status])[0]
    occupation = label_encoders['Ocupacion'].transform([occupation])[0]
    relationship = label_encoders['Relacion'].transform([relationship])[0]
    race = label_encoders['Raza'].transform([race])[0]
    sex = label_encoders['Genero'].transform([sex])[0]
    native_country = label_encoders['Pais'].transform([native_country])[0]
    
    # Crear el array de entrada
    input_data = np.array([[age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week, workclass, education, marital_status, occupation, relationship, race, sex, native_country]])
    input_data_scaled = scaler.fit_transform(input_data)  # Aquí, usamos fit_transform solo para la demo, usar el escalador original del entrenamiento
    
    # Hacer la predicción
    prediction = model.predict(input_data_scaled)
    return prediction[0]

# Título de la aplicación
st.title("Predicción de Ingresos")

# Entradas del usuario
age = st.number_input("Edad", min_value=0, max_value=100, value=30)
fnlwgt = st.number_input("Ingresos", min_value=0, value=1000)
education_num = st.number_input("Número de Educación", min_value=0, max_value=20, value=10)
capital_gain = st.number_input("Ganancia de Capital", min_value=0, value=0)
capital_loss = st.number_input("Pérdida de Capital", min_value=0, value=0)
hours_per_week = st.number_input("Horas por Semana", min_value=0, max_value=100, value=40)

# Selectboxes para variables categóricas
workclass = st.selectbox("Clase de Trabajo", workclass_options)
education = st.selectbox("Educación", education_options)
marital_status = st.selectbox("Estado Civil", marital_status_options)
occupation = st.selectbox("Ocupación", occupation_options)
relationship = st.selectbox("Relación", relationship_options)
race = st.selectbox("Raza", race_options)
sex = st.selectbox("Sexo", sex_options)
native_country = st.selectbox("País de Origen", native_country_options)

# Botón para hacer la predicción
if st.button("Predecir"):
    result = predict_income(age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week, workclass, education, marital_status, occupation, relationship, race, sex, native_country)
    st.write(f"El ingreso predicho es: {'<=50K' if result == 0 else '>50K'}")
    st.write(f"Resultado = {result}")

