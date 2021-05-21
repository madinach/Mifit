import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
from scipy import stats

# import category_encoders as ce
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from sklearn import pipeline      # Pipeline
from sklearn import preprocessing  # OrdinalEncoder, LabelEncoder
from sklearn import impute
from sklearn import compose
from sklearn import model_selection  # train_test_split
# accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn import metrics
from sklearn import set_config
from sklearn.datasets import make_classification
# from feature_engine.selection import SmartCorrelatedSelection

from PIL import Image
import pickle


# Load Data

@st.cache
def load_data(filename=None):
    filename_default = './data/heart.csv'
    if not filename:
        filename = filename_default

    df = pd.read_csv(f"./{filename}")
    return df
    # return df, df.shape[0], df.shape[1], filename


data = load_data()

load_clf = pickle.load(open('./data/model.pkl', "rb"))


##################################################################################

##### Plots #####


##################################################################################


def main():
    menu = ["Home", "About", "Concept", "Calculator"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":

        ####################################################
        header = st.beta_container()
        team = st.beta_container()
        activities = st.beta_container()
        github = st.beta_container()
        # dataset = st.beta_container()
        # conclusion = st.beta_container()
        # footer = st.beta_container()
        ####################################################
        with header:
            st.title('MIFit model predictions')  # site title h1
            st.markdown("""---""")
            st.header('Welcome to our Machine Learning Project!')
            st.text(' ')
            image = Image.open('./data/logo.png')
            st.image(image, caption="")

            with team:
                # meet the team button
                st.sidebar.subheader('John Locke Team')

                st.sidebar.markdown(
                    '[Madina Zhenisbek](https://github.com/)')
                st.sidebar.markdown(
                    '[Hedaya Ali](https://github.com/madinach)')
                st.sidebar.markdown(
                    '[Thomas](https://github.com/)')
                st.sidebar.markdown(
                    '[Paramveer](https://github.com/)')

                st.sidebar.text(' ')
                st.sidebar.text(' ')

        with github:
            # github section:
            st.header('GitHub / Instructions')
            st.markdown(
                'Check the instruction [here](https://github.com/madinach/Mifit/blob/main/README.md)')
            st.text(' ')


##########################################################################
    elif choice == "Data Analysis":
        st.subheader("Data Analysis")
        # data = load_data('raw')
        #header = st.beta_container()
        #dataset = st.beta_container()
'''
        with dataset:
            st.title("About")

            #### Data Correlation ####
            st.set_option('deprecation.showPyplotGlobalUse', False)

            st.text('Data Correlation ')
            sns.set(style="white")
            plt.rcParams['figure.figsize'] = (15, 10)
            sns.heatmap(data.corr(), annot=True, linewidths=.5, cmap="Blues")
            plt.title('Corelation Between Variables', fontsize=30)
            plt.show()
            st.pyplot()

            #### Box Plot #####
            st.text('Outlier Detection ')
            fig = plt.figure(figsize=(15, 10))
            sns.boxplot(data=data)
            st.pyplot(fig)

  

            st.text(' ')
            st.header("Variables or features explanations:")
            st.markdown("""* age (Age in years)""")
            st.markdown("""* sex : (1 = male, 0 = female)""")
            st.markdown(
                """* cp (Chest Pain Type): [0: Typical Angina, 1: Atypical Angina, 2: Non-Anginal Pain, 3: Asymptomatic]""")
            st.markdown("""* trestbps (Resting Blood Pressure in mm/hg )""")
            st.markdown("""* chol (Serum Cholesterol in mg/dl)""")
            st.markdown(
                """* fps (Fasting Blood Sugar > 120 mg/dl): [0 = no, 1 = yes]""")
            st.markdown(
                """* restecg (Resting ECG): [0: normal, 1: having ST-T wave abnormality , 2: showing probable or definite left ventricular hypertrophy]""")
            st.markdown("""* thalach (maximum heart rate achieved)""")
            st.markdown(
                """* exang (Exercise Induced Angina): [1 = yes, 0 = no]""")
            st.markdown(
                """* oldpeak (ST depression induced by exercise relative to rest)""")
            st.markdown(
                """* slope (the slope of the peak exercise ST segment)""")
            st.markdown("""* ca [number of major vessels (0–3)]""")
            st.markdown(
                """* thal : [1 = normal, 2 = fixed defect, 3 = reversible defect]""")
            st.markdown("""* target: [0 = disease, 1 = no disease]""")

    elif choice == "ML":


        with footer:
            # Footer
            st.markdown("""---""")
            st.markdown("Heart Attack Predictions - Machine Learning Project")
            st.markdown("")
            st.markdown(
                "If you have any questions, checkout our [documentation](https://github.com/fistadev/starwars_data_project) ")
            st.text(' ')

        ############################################################################################################################
    else:
        st.header("Predictions")

        def xgb_page_builder(data):
            st.sidebar.header('Heart Attack Predictions')
            st.sidebar.markdown('You can tune the parameters by siding')
            st.sidebar.text_input("What's your age?")
            
            cp = st.sidebar.slider(
                'Select max_depth (default = 30)', 0, 1, 2)
            thalach = st.sidebar.slider(
                'Select learning rate (divided by 10) (default = 0.1)', min_value=50 , max_value=300 , value=None , step=5)
            slope = st.sidebar.slider(
                'Select min_child_weight (default = 0.3)', 1, 2, 3)


        #st.write(xgb_page_builder(data))
        st.sidebar.header('Heart Attack Predictions')

        st.text(' ')
        st.markdown('Model selection')
        st.text(' ')
        #image = Image.open('./data/model-selection.png')
        #st.image(image, caption="")
        st.text(' ')

        st.text(' ')
        st.markdown('Selecting the best model with KFold')
        st.text(' ')
        #image = Image.open('./data/kfold.png')
        #st.image(image, caption="")
        st.text(' ')

"""

        set_config(display='diagram')
        st.write(load_clf)
        a = [
            54,
            1, 
            cp,
            131,
            246,
            0.148,
            5280, 
            thalch, 
            0.326, 
            1.0396,
            slope,
            0.7293,
            2.313, 
            ]
        preds = load_clf.predict(a)
        print(preds)
'''
main()
