import streamlit as st
import requests

# Set the FastAPI endpoint URL
api_url = "http://127.0.0.1:8000/classifier/"

# Streamlit interface
st.title("Hate Speech Detection")

# Input text box
user_input = st.text_area("Enter a sentence:", "")

# Validate button
if st.button("Validate"):
    if user_input:
        try:
            # Call the FastAPI API
            response = requests.post(api_url, json={"text": user_input})
            
            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()
                st.write("Output")
                print(f"Speech: {result['text']}")
                print(f"Label: {result['Label']}")
                print(f"Score: {result['score']}")
               
                st.write(f"Speech: {result['text']}")
                st.write(f"Label: {result['Label']}")
                st.write(f"Score: {result['score']}")
                
            else:
                # Display error status and response content
                st.write(f"Error: Received status code {response.status_code}")
                st.write(f"Response content: {response.text}")
        except Exception as e:
            # Handle any exceptions during the request
            st.write(f"Request failed: {e}")
    else:
        st.write("Please enter a sentence to validate.")