import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import pandas as pd
from sklearn.ensemble._forest import ForestClassifier, ForestRegressor
import pickle
from PIL import Image
import tensorflow as tf
from streamlit_option_menu import option_menu
import os

def predict_malaria(img):
    img = img.resize((36,36))
    img = np.asarray(img)
    img = img.reshape((1,36,36,3))
    img = img.astype(np.float64)
    model = load_model("malaria.h5")
    pred_probs = model.predict(img)[0]
    pred_class = np.argmax(pred_probs)
    pred_prob = pred_probs[pred_class]
    return pred_class, pred_prob

def predict_pneumonia(img):
    img = img.convert('L')
    img = img.resize((36,36))
    img = np.asarray(img)
    img = img.reshape((1,36,36,1))
    img = img / 255.0
    model = load_model("pneumonia.h5")
    pred_probs = model.predict(img)[0]
    pred_class = np.argmax(pred_probs)
    pred_prob = pred_probs[pred_class]
    return pred_class, pred_prob


with open('diabetes_model.sav', 'rb') as file:
    diabetes_model = pickle.load(file)

#logo = Image.open("/Users/thanizeassuncaorodrigues/Documents/GitHub/DiagnoSys/logo/MediScan.png")
# Criação de uma sidebar personalizada com ícones redondos
#st.sidebar.image(logo, use_column_width=True)
st.sidebar.title("Menu")

menu = st.sidebar.radio(
    "Navegação",
    ["🏠 Página Inicial", "🦟 Detecção Malaria", " 🫁 Detecção Pneumonia", "💉 Detecção Diabetes", "📊 Datasets Disponíveis"]
)
# Função para mapear seleção de menu para página correspondente
def get_selected_page(menu):
    if menu == "🏠 Página Inicial":
        return "home"
    elif menu == "🦟 Detecção Malaria":
        return "Malaria"
    elif menu == " 🫁 Detecção Pneumonia":
        return "Pneumonia"
    elif menu == "💉 Detecção Diabetes":
        return "Diabetes"
    elif menu == "📊 Datasets Disponíveis":
        return "Datasets"
    
selected_page = get_selected_page(menu)


def main(selected_page):
    # Conteúdo da página selecionada
    if selected_page == "home":
        st.title('Bem-vindo à Aplicação de Previsão de Anomalias Médicas')
        st.write("Este é um projeto de previsão de diversas anomalias médicas usando modelos de deep learning e machine learning.")

        st.write("É importante observar que os modelos utilizados nesta aplicação foram obtidos de repositórios públicos na internet e, portanto, sua confiabilidade pode variar.")

        st.write("Embora tenham sido treinados em grandes conjuntos de dados médicos, é fundamental lembrar que todas as previsões devem ser verificadas por profissionais de saúde qualificados.")

        # Seção de Perguntas Frequentes
        st.subheader("Perguntas Frequentes")

        # Lista de perguntas frequentes e respostas
        faq = [
            {
                "pergunta": "Como a previsão de anomalias é feita?",
                "resposta": "A detecção de pneumonia e malária é feita usando uma rede neural convolucional (CNN), enquanto a seção de diabetes é detectada por um modelo Random Forest",
            },
            {
                "pergunta": "Os modelos são precisos?",
                "resposta": "Os modelos foram treinados em grandes conjuntos de dados médicos, mas lembre-se de que todas as previsões devem ser verificadas por profissionais de saúde qualificados.",
            },
            {
                "pergunta": "Qual é o propósito desta aplicação?",
                "resposta": "Esta aplicação foi desenvolvida para auxiliar na detecção de diversas anomalias médicas em imagens de diferentes partes do corpo.",
            },
            {
                "pergunta": "Quais tipos de anomalias médicas podem ser detectadas?",
                "resposta": "Os modelos podem detectar várias anomalias, incluindo pneumonia, malária e diabetes.",
            },
            
        ]

        # Exibição das perguntas frequentes
        for item in faq:
            with st.expander(item["pergunta"]):
                st.write(item["resposta"])


    elif selected_page ==  "Malaria":
        st.header("Previsão de Malária")
        uploaded_file = st.file_uploader("Faça o upload de uma imagem para previsão de malária", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file)
                st.image(img, caption="Imagem enviada", use_column_width=True)
                pred_class, pred_prob = predict_malaria(img)
                
                if pred_class == 1:
                    st.write("Previsão: Infectado")
                    st.write(f"Probabilidade de Malária: {pred_prob * 100:.2f}%")
                else:
                    st.write("Previsão: Não está infectado")
                    st.write(f"Probabilidade de Saúde: {pred_prob * 100:.2f}%")
                    
            except Exception as e:
                st.error(f"Erro ao prever Malária: {str(e)}")

    elif selected_page ==  "Pneumonia":
        st.header("Previsão de Pneumonia")
        uploaded_file = st.file_uploader("Faça o upload de uma imagem para previsão de pneumonia", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file)
                st.image(img, caption="Imagem enviada", use_column_width=True)
                pred_class, pred_prob = predict_pneumonia(img)
                
                if pred_class == 1:
                    st.write("Previsão: Pneumonia")
                    st.write(f"Probabilidade de Pneumonia: {pred_prob * 100:.2f}%")
                else:
                    st.write("Previsão: Saudável")
                    st.write(f"Probabilidade de Saúde: {pred_prob * 100:.2f}%")
                    
            except Exception as e:
                st.error(f"Erro ao prever Pneumonia: {str(e)}")

    elif selected_page ==  "Diabetes":
        # Título da página
        st.title('Previsão de Diabetes')

        # obtendo os dados de entrada do usuário
        col1, col2, col3 = st.columns(3)

        with col1:
            Pregnancies = st.text_input('Número de Gestações')

        with col2:
            Glucose = st.text_input('Nível de Glicose')

        with col3:
            BloodPressure = st.text_input('Valor da Pressão Arterial')

        with col1:
            SkinThickness = st.text_input('Valor da Espessura da Pele')

        with col2:
            Insulin = st.text_input('Nível de Insulina')

        with col3:
            BMI = st.text_input('Valor do IMC')

        with col1:
            DiabetesPedigreeFunction = st.text_input('Valor da Função de Pedigree de Diabetes')

        with col2:
            Age = st.text_input('Idade da Pessoa')


        # código para previsão
        diab_diagnosis = ''

        # criando um botão para previsão

        if st.button('Resultado do Teste de Diabetes'):

            user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                        BMI, DiabetesPedigreeFunction, Age]

            user_input = [float(x) for x in user_input]

            diab_prediction = diabetes_model.predict([user_input])

            if diab_prediction[0] == 1:
                diab_diagnosis = 'A pessoa é diabética'
            else:
                diab_diagnosis = 'A pessoa não é diabética'

        st.success(diab_diagnosis)

    elif selected_page == "Datasets":
        # Título da página
        st.title('Datasets Disponíveis')
        # Introdução
        st.write("Esta página contém links para download e visualização de datasets utilizados na aplicação.")
        
        # Dicionário com URLs dos datasets
        datasets = {
            "Dataset de Malária": "https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria",
            "Dataset de Pneumonia": "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia",
            "Dataset de Doenças Cardíacas": "https://github.com/siddhardhan23/multiple-disease-prediction-streamlit-app/blob/main/dataset/heart.csv",
            "Dataset de Doenças Renais": "https://www.kaggle.com/datasets/mansoordaku/ckdisease",
            "Dataset de Diabetes": "https://github.com/siddhardhan23/multiple-disease-prediction-streamlit-app/blob/main/dataset/diabetes.csv",
            "Dataset de Doenças Hepáticas": "https://www.kaggle.com/datasets/uciml/indian-liver-patient-records",
            "Dataset de Câncer de Mama": "https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data"
        }

        # Loop sobre os datasets para exibir links de download e botões para visualização
        for dataset_name, dataset_url in datasets.items():
            st.write(f"**{dataset_name}:**")
            st.markdown(f"[Download {dataset_name}]({dataset_url})")

if __name__ == "__main__":
    main(selected_page)
