import streamlit as st
import joblib
import os


def query_workspace():
    working_dir = st.sidebar.text_input('Enter the prior B-SOiD working directory:')
    try:
        os.listdir(working_dir)
        st.markdown(
            'You have selected **{}** for B-SOiD output directory.'.format(working_dir))
    except FileNotFoundError:
        st.error('No such directory')
    files = [i for i in os.listdir(working_dir) if os.path.isfile(os.path.join(working_dir, i)) and \
             '_predictions.sav' in i and not '_accuracy' in i and not '_coherence' in i]
    bsoid_variables = [files[i].partition('_predictions.sav')[0] for i in range(len(files))]
    bsoid_prefix = []
    for var in bsoid_variables:
        if var not in bsoid_prefix:
            bsoid_prefix.append(var)
    prefix = st.selectbox('Select prior B-SOiD prefix', bsoid_prefix)
    try:
        st.markdown('You have selected **{}_XXX.sav** for prior prefix.'.format(prefix))
    except TypeError:
        st.error('Please input a prior prefix to load workspace.')
    return working_dir, prefix


@st.cache
def load_predictions(path, name):
    with open(os.path.join(path, str.join('', (name, '_predictions.sav'))), 'rb') as fr:
        data = joblib.load(fr)
    return [i for i in data]