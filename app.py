import streamlit as st
import pickle
import json

# Load your 3 files
scaler = pickle.load(open('scaler.pkl','rb'))
kmeans = pickle.load(open('kmeans.pkl','rb'))
mapping = json.load(open('mapping.json','r'))
mapping = {int(k): v for k, v in mapping.items()} # fix keys

st.title("Customer Segmentation - 3 Groups")

income = st.number_input("Annual Income (k$)", 15, 150, 60)
spending = st.number_input("Spending Score (1-100)", 1, 100, 50)

if st.button("Predict Group"):
    scaled = scaler.transform([[income, spending]])
    cluster_id = kmeans.predict(scaled)[0]
    result = mapping[cluster_id]

    st.success(f"**{result}**")

    if "Low" in result:
        st.warning("Strategy: Give Discounts")
    elif "Medium" in result:
        st.info("Strategy: Show New Products")
    else:
        st.balloons()
        st.success("Strategy: VIP - Premium Offers")