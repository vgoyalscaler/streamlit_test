#!/usr/bin/env python
# coding: utf-8

# ### Importing all data for data pre-processing

# In[1]:


# importing null handler data
import pandas as pd
import pickle
imputer_dict = pickle.load(open('nma_lr_imputer_dict.pkl', 'rb'))
first_three_avg_rating_super_batch_mean = imputer_dict['first_three_avg_rating_super_batch_mean']
first_three_avg_rating_overall_mean = imputer_dict['first_three_avg_rating_overall_mean']
first_3_total_assignment_problems_solved_ratio_super_batch_mean = imputer_dict['first_3_total_assignment_problems_solved_ratio_super_batch_mean']
first_3_total_assignment_problems_solved_ratio_overall_mean = imputer_dict['first_3_total_assignment_problems_solved_ratio_overall_mean']

# importing target encodign data
target_encodings_dict = pickle.load(open('nma_lr_target_encodings_dict.pkl', 'rb'))
super_batch_name_target_encoded = target_encodings_dict['super_batch_name_target_encoded']
module_name_target_encoded = target_encodings_dict['module_name_target_encoded']
next_module_name_target_encoded = target_encodings_dict['next_module_name_target_encoded']
instructor_email_target_encoded = target_encodings_dict['instructor_email_target_encoded']
next_module_instructor_email_target_encoded = target_encodings_dict['next_module_instructor_email_target_encoded']
overall_y_mean = target_encodings_dict['overall_y_mean']

# Importing One Hot Encoding data
ohe_columns = pickle.load(open('nma_lr_ohe_columns.pkl', 'rb'))
 
# Importing Standardization data
scaler = pickle.load(open('nma_lr_scaler.pkl', 'rb'))

# Importing main model
train_results = pickle.load(open('nma_lr_model.pkl', 'rb'))


# ### Defining all required functions for data pre-processing

# In[2]:


# null_handler function
def null_handler(df):
    df['first_three_avg_rating'] = df.apply(
        lambda row: first_three_avg_rating_super_batch_mean[row['super_batch_name']] if pd.isna(row['first_three_avg_rating']) and row['super_batch_name'] in first_three_avg_rating_super_batch_mean.index
                    else first_three_avg_rating_overall_mean if pd.isna(row['first_three_avg_rating'])
                    else row['first_three_avg_rating'],
        axis=1
    )
    
    df['first_3_total_assignment_problems_solved_ratio'] = df.apply(
        lambda row: first_3_total_assignment_problems_solved_ratio_super_batch_mean[row['super_batch_name']] if pd.isna(row['first_3_total_assignment_problems_solved_ratio']) and row['super_batch_name'] in first_3_total_assignment_problems_solved_ratio_super_batch_mean.index
                    else first_3_total_assignment_problems_solved_ratio_overall_mean if pd.isna(row['first_3_total_assignment_problems_solved_ratio'])
                    else row['first_3_total_assignment_problems_solved_ratio'],
        axis=1
    )
    
    return df

# target encoding function
def train_target_encodings(df):
    # Handle super_batch_name encoding
    df['super_batch_name'] = df['super_batch_name'].map(super_batch_name_target_encoded).fillna(overall_y_mean)

    # Handle next_module_name encoding
    df['next_module_name'] = df['next_module_name'].map(next_module_name_target_encoded).fillna(overall_y_mean)

    # Handle instructor_email encoding
    df['instructor_email'] = df['instructor_email'].map(instructor_email_target_encoded).fillna(overall_y_mean)

    return df

# one hot encoding function
def one_hot_encoding_train(df, ohe_columns):
    # one hot encoding (OHE)
    track_status = pd.get_dummies(df['track'],drop_first=True)

    # ohe_columns = track_status.columns
    track_status = track_status.reindex(columns=ohe_columns, fill_value=0)

    df = pd.concat([df, track_status], axis=1)
    df.drop('track', axis=1, inplace=True)
    
    return df

# standardization function
def standardization_train(df, scaler):
    df1 = df.drop(['Beginner', 'Intermediate', 'mbe_eligible_flag'], axis=1)
    df2 = df[['Beginner', 'Intermediate', 'mbe_eligible_flag']]
    
    df1_scaled = scaler.transform(df1)
    df1_scaled = pd.DataFrame(df1_scaled, columns=df1.columns)
    
    df2.reset_index(drop=True, inplace=True)
    df = pd.concat([df1_scaled,df2], axis=1)
    
    return df



# ### Calling streamlit function and running the main prediction code

# In[3]:


import streamlit as st 

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
            df_initial = df.copy()
            df = null_handler(df) # null handling
            df = train_target_encodings(df) # target encodings
            df = one_hot_encoding_train(df, ohe_columns) # one-hot encoding
            df = standardization_train(df, scaler) # standardization
            
            import statsmodels.api as sm
            x_sm = sm.add_constant(df)

            y_pred = train_results.predict(x_sm)
            y_pred = y_pred.rename('next_module_attendance_ratio_per_eligible_learners')

            df = pd.concat([df_initial,y_pred], axis=1)
            
            output_csv = convert_df_to_csv(df)
            st.write('Output Data:')
            st.write(df)

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


# In[4]:


# !jupyter nbconvert --to script streamlit_execution_test.ipynb


# In[18]:


# !streamlit run streamlit_execution_test.py


# In[38]:


# import pandas as pd
# ama_test_df = pd.read_csv('ama_test.csv')
# ama_test_df_initial = ama_test_df.copy()
# ama_test_df


# In[39]:


# import pickle
# imputer_dict = pickle.load(open('nma_lr_imputer_dict.pkl', 'rb'))
# imputer_dict


# In[40]:


# first_three_avg_rating_super_batch_mean = imputer_dict['first_three_avg_rating_super_batch_mean']
# first_three_avg_rating_overall_mean = imputer_dict['first_three_avg_rating_overall_mean']
# first_3_total_assignment_problems_solved_ratio_super_batch_mean = imputer_dict['first_3_total_assignment_problems_solved_ratio_super_batch_mean']
# first_3_total_assignment_problems_solved_ratio_overall_mean = imputer_dict['first_3_total_assignment_problems_solved_ratio_overall_mean']


# In[41]:


# def null_handler(df):
#     df['first_three_avg_rating'] = df.apply(
#         lambda row: first_three_avg_rating_super_batch_mean[row['super_batch_name']] if pd.isna(row['first_three_avg_rating']) and row['super_batch_name'] in first_three_avg_rating_super_batch_mean.index
#                     else first_three_avg_rating_overall_mean if pd.isna(row['first_three_avg_rating'])
#                     else row['first_three_avg_rating'],
#         axis=1
#     )
    
#     df['first_3_total_assignment_problems_solved_ratio'] = df.apply(
#         lambda row: first_3_total_assignment_problems_solved_ratio_super_batch_mean[row['super_batch_name']] if pd.isna(row['first_3_total_assignment_problems_solved_ratio']) and row['super_batch_name'] in first_3_total_assignment_problems_solved_ratio_super_batch_mean.index
#                     else first_3_total_assignment_problems_solved_ratio_overall_mean if pd.isna(row['first_3_total_assignment_problems_solved_ratio'])
#                     else row['first_3_total_assignment_problems_solved_ratio'],
#         axis=1
#     )
    
#     return df
    


# In[42]:


# ama_test_df = null_handler(ama_test_df)
# ama_test_df


# ### Encodings

# Target Encoding

# In[43]:


# target_encodings_dict = pickle.load(open('nma_lr_target_encodings_dict.pkl', 'rb'))
# target_encodings_dict


# In[44]:


# super_batch_name_target_encoded = target_encodings_dict['super_batch_name_target_encoded']
# module_name_target_encoded = target_encodings_dict['module_name_target_encoded']
# next_module_name_target_encoded = target_encodings_dict['next_module_name_target_encoded']
# instructor_email_target_encoded = target_encodings_dict['instructor_email_target_encoded']
# next_module_instructor_email_target_encoded = target_encodings_dict['next_module_instructor_email_target_encoded']
# overall_y_mean = target_encodings_dict['overall_y_mean']
 


# In[45]:


# def train_target_encodings(df):
#     # Handle super_batch_name encoding
#     df['super_batch_name'] = df['super_batch_name'].map(super_batch_name_target_encoded).fillna(overall_y_mean)

#     # Handle next_module_name encoding
#     df['next_module_name'] = df['next_module_name'].map(next_module_name_target_encoded).fillna(overall_y_mean)

#     # Handle instructor_email encoding
#     df['instructor_email'] = df['instructor_email'].map(instructor_email_target_encoded).fillna(overall_y_mean)

#     return df


# In[46]:


# ama_test_df = train_target_encodings(ama_test_df)
# ama_test_df


# One Hot Encoding

# In[47]:


# ohe_columns = pickle.load(open('nma_lr_ohe_columns.pkl', 'rb'))
# ohe_columns


# In[48]:


# def one_hot_encoding_train(df, ohe_columns):
#     # one hot encoding (OHE)
#     track_status = pd.get_dummies(df['track'],drop_first=True)

#     # ohe_columns = track_status.columns
#     track_status = track_status.reindex(columns=ohe_columns, fill_value=0)

#     df = pd.concat([df, track_status], axis=1)
#     df.drop('track', axis=1, inplace=True)
    
#     return df


# In[49]:


# ama_test_df = one_hot_encoding_train(ama_test_df, ohe_columns)
# ama_test_df


# ### Doing Standardization

# In[50]:


# scaler = pickle.load(open('nma_lr_scaler.pkl', 'rb'))


# In[51]:


# def standardization_train(df, scaler):
#     df1 = df.drop(['Beginner', 'Intermediate', 'mbe_eligible_flag'], axis=1)
#     df2 = df[['Beginner', 'Intermediate', 'mbe_eligible_flag']]
    
#     df1_scaled = scaler.transform(df1)
#     df1_scaled = pd.DataFrame(df1_scaled, columns=df1.columns)
    
#     df2.reset_index(drop=True, inplace=True)
#     df = pd.concat([df1_scaled,df2], axis=1)
    
#     return df
    


# In[52]:


# ama_test_df = standardization_train(ama_test_df, scaler)
# ama_test_df


# ### Inputting data in the model

# In[53]:


# train_results = pickle.load(open('nma_lr_model.pkl', 'rb'))


# In[58]:


# import statsmodels.api as sm
# x_sm = sm.add_constant(ama_test_df)

# y_pred = train_results.predict(x_sm)
# y_pred = y_pred.rename('next_module_attendance_ratio_per_eligible_learners')
# y_pred


# In[60]:


# final_df = pd.concat([ama_test_df_initial,y_pred], axis=1)
# final_df


# In[61]:


# final_df.to_csv('ama_test_output.csv')


# In[ ]:




