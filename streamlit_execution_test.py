#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st 

def main():
    st.title('Just Testing')
    batch = st.text_input('Enter the super_batch_name')
    
    if st.button('Predict'):
        st.write('hip-hip horrayy!')
        
if __name__ == '__main__':
    main()


# In[ ]:




