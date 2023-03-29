#%%load package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import shap
import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

#%%不提示warning信息
st.set_option('deprecation.showPyplotGlobalUse', False)

#%%set title
st.set_page_config(page_title='Machine learning predictive model of eye metastasis of nasopharyngeal carcinoma: Based on AdaBoost method')
st.title('Machine learning predictive model of eye metastasis of nasopharyngeal carcinoma: Based on AdaBoost method')

#%%set variables selection
st.sidebar.markdown('## Variables')
Pathology_type = st.sidebar.selectbox('Pathology type',('Squamous cell carcinoma','Undifferentiated carcinoma'),index=1)
Hb = st.sidebar.slider("Hb(g/L)", 0, 200, value=110, step=1)
TG = st.sidebar.slider("TG(mmol/L)", 0.00,10.00, value=2.42, step=0.01)
TC = st.sidebar.slider("TC(mmol/L)", 0.00,10.00, value=1.42, step=0.01)
CA199 = st.sidebar.slider("CA199(μg/L)", 0.00, 500.00, value=59.61, step=0.01)
Cyfra_21_1 = st.sidebar.slider("Cyfra 21-1(μg/L)", 0.00, 100.00, value=9.61, step=0.01)


#分割符号
st.sidebar.markdown('#  ')
st.sidebar.markdown('#  ')
st.sidebar.markdown('##### All rights reserved') 
st.sidebar.markdown('##### For communication and cooperation, please contact wshinana99@163.com, Wu Shi-Nan, Nanchang university')
#传入数据
map = {'Squamous cell carcinoma':0,'Undifferentiated carcinoma':1}
Pathology_type =map[Pathology_type]
# 数据读取，特征标注
#%%load model
ab_model = joblib.load('ab_biyanai_model.pkl')

#%%load data
hp_train = pd.read_csv('github_biyanai_data.csv')
features =["Pathology_type","Hb","TG","TC","CA199",'Cyfra_21_1']
target = 'M'
y = np.array(hp_train[target])
sp = 0.5

is_t = (ab_model.predict_proba(np.array([[Pathology_type,Hb,TG,TC,CA199,Cyfra_21_1]]))[0][1])> sp
prob = (ab_model.predict_proba(np.array([[Pathology_type,Hb,TG,TC,CA199,Cyfra_21_1]]))[0][1])*1000//1/10
    

if is_t:
    result = 'High Risk Ocular metastasis'
else:
    result = 'Low Risk Ocular metastasis'
if st.button('Predict'):
    st.markdown('## Result:  '+str(result))
    if result == '  Low Risk Ocular metastasis':
        st.balloons()
    st.markdown('## Probability of High Risk Ocular metastasis group:  '+str(prob)+'%')
    #%%cbind users data
    col_names = features
    X_last = pd.DataFrame(np.array([[Pathology_type,Hb,TG,TC,CA199,Cyfra_21_1]]))
    X_last.columns = col_names
    X_raw = hp_train[features]
    X = pd.concat([X_raw,X_last],ignore_index=True)
    if is_t:
        y_last = 1
    else:
        y_last = 0  
    y_raw = (np.array(hp_train[target]))
    y = np.append(y_raw,y_last)
    y = pd.DataFrame(y)
    model = ab_model
    #%%calculate shap values
    sns.set()
    explainer = shap.Explainer(model, X)
    shap_values = explainer.shap_values(X)
    a = len(X)-1
    #%%SHAP Force logit plot
    st.subheader('SHAP Force logit plot of AdaBoost model')
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    force_plot = shap.force_plot(explainer.expected_value,
                    shap_values[a, :], 
                    X.iloc[a, :], 
                    figsize=(25, 3),
                    # link = "logit",
                    matplotlib=True,
                    out_names = "Output value")
    st.pyplot(force_plot)
    #%%SHAP Water PLOT
    st.subheader('SHAP Water plot of AdaBoost model')
    shap_values = explainer(X) # 传入特征矩阵X，计算SHAP值
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    waterfall_plot = shap.plots.waterfall(shap_values[a,:])
    st.pyplot(waterfall_plot)
    #%%ConfusionMatrix 
    st.subheader('Confusion Matrix of AdaBoost model')
    ab_prob = ab_model.predict(X)
    cm = confusion_matrix(y, ab_prob)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NOM', 'OM'])
    sns.set_style("white")
    disp.plot(cmap='RdPu')
    plt.title("Confusion Matrix of AdaBoost model")
    disp1 = plt.show()
    st.pyplot(disp1)

