#!/usr/bin/env python
# coding: utf-8

# In[67]:


import streamlit as st 
import pandas as pd

def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

def main():
    st.title("Predicting next module attendance ratio at super_batch level wrt batch's eligible learners for next module")
    
    # take csv file as input 
    uploaded_csv = st.file_uploader("Upload the csv file of the required batches and corresponding next_module_names selected from: https://docs.google.com/spreadsheets/d/1gCImAFX1TVeKCSRDCZrIcJWWpKSbAZaFwljqVlWQgxs/edit?gid=0#gid=0", type=['csv'])
    
    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.write("Data Uploaded:")
        st.write(df)
      
        if st.button('Predict'): 
            output_csv = convert_df_to_csv(df)

            st.download_button(
                label = 'Download Output CSV',
                data = output_csv,
                file_name = 'output_csv.csv',
                mime = 'text/csv'
            )
        
    else:
        st.warning('Please upload a csv file to proceed')
if __name__ == '__main__':
    main()


# In[68]:


get_ipython().system('jupyter nbconvert --to script streamlit_execution_test.ipynb')


# In[18]:


# !streamlit run streamlit_execution_test.py

