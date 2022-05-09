import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.title('Linen Assesment Passed Level 1 Analysis and Prediction')
# load dataset
df = pd.read_csv('linenassesment.csv')

# show the entire dataframe
st.write(df)

# f-string
st.subheader('Passed Level Rate for All')
passed_count = df['Passed'].value_counts()
st.text(f'Passed Level rate = {passed_count.values[1]/sum(passed_count):.2%}')

# simple plotting
st.subheader('Passed Level Rate by Age ')
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
passed_count.plot.bar(ax=ax[0])
df['Age'].plot.hist(ax=ax[1])
st.pyplot(fig)

# simple plotting
st.subheader('Passed Level by Placement Test ')
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
passed_count.plot.bar(ax=ax[0])
df['Placetest'].plot.hist(ax=ax[1])
st.pyplot(fig)

# simple plotting
st.subheader('Passed Level by Education ')
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
passed_count.plot.bar(ax=ax[0])
df['Education'].plot.hist(ax=ax[1])
st.pyplot(fig)

st.subheader('Passed Level Rate by Number of Siblings')
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
passed_count.plot.bar(ax=ax[0])
df['Sibling'].plot.hist(ax=ax[1])
st.pyplot(fig)

st.subheader('Passed Level Rate by Number of Parents')
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
passed_count.plot.bar(ax=ax[0])
df['Parent'].plot.hist(ax=ax[1])
st.pyplot(fig)


st.subheader('Making Prediction')
st.markdown('**Please provide User information**:')  # you can use markdown like this


# load models
tree_clf = joblib.load('clf-linen-best.pickle')

# get inputs

Sex = st.selectbox('Sex', ['female', 'male'])
Age = int(st.number_input('Age:', 0, 120, 20))
Sibling = int(st.number_input('# of siblings :', 0, 10, 0))
Parent = int(st.number_input('# of parents :', 0, 2, 0))
Education = st.selectbox('Education Below (1 = Under Graduate, 2 = Graduate, 3 = Post Graduate)', [1, 2, 3])
Placetest = int(st.number_input('# of placement test result:', 0, 100, 0))
English = st.selectbox('English Qualification (B= Beginner, I = Intermediate, A = Advanced)', ['B', 'I', 'A'])

# this is how to dynamically change text
prediction_state = st.markdown('calculating...')

user = pd.DataFrame(
    {
        'Education': [Education],
        'Sex': [Sex],
        'Age': [Age],
        'Sibling': [Sibling],
        'Parent': [Parent],
        'Placetest': [Placetest],
        'English': [English],
    }
)

y_pred = tree_clf.predict(user)

if y_pred[0] == 0:
    msg = 'This user is predicted to be: **not passed Level 1 **'
else:
    msg = 'This user is predicted to be: **Passed Level 1**'

prediction_state.markdown(msg)
