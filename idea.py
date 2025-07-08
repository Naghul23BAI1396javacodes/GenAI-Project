import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
import requests
from fpdf import FPDF
import tempfile
from datetime import datetime

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("YOUR_GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

st.set_page_config(page_title="Travel Planner", page_icon="🧳", layout="centered")
st.title("🧭 Travel Planner")
st.write("Plan your perfect trip based on destination, dates, interests, transport, budget, and more!")

# Inputs
start_location = st.text_input("Your Starting Location", placeholder="example: Chennai, Delhi")
destination = st.text_input("Destination", placeholder="example: Goa, Paris")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date")
with col2:
    end_date = st.date_input("End Date")

trip_days = (end_date - start_date).days
if trip_days <= 0:
    st.warning("Please ensure end date is after the start date.")

min_budget = st.number_input("Minimum Budget (in ₹)", min_value=1000, step=500)
max_budget = st.number_input("Maximum Budget (in ₹)", min_value=min_budget, step=500)

people = st.number_input("Number of People in Trip", min_value=1, step=1)

transport_mode = st.selectbox(
    "Preferred Mode of Transport (between cities/states)",
    ["Train", "Car", "Aeroplane", "Ship", "Bus", "Other"]
)

food_type = st.radio("Preferred Food Type", ["Vegetarian", "Non-Vegetarian"])

# Preferences
default_preferences = [
    "Beach", "Food", "Mountains", "Shopping", "Culture",
    "Adventure", "Nightlife", "Nature", "Relaxation", "Other"
]
selected_preferences = st.multiselect("Choose Your Preferred Outing Types", default_preferences)

custom_preferences = []
if "Other" in selected_preferences:
    custom_input = st.text_input("Add your own outing types (press Enter to save)", key="custom")
    if custom_input and custom_input not in custom_preferences:
        custom_preferences.append(custom_input)

final_preferences = [p for p in selected_preferences if p != "Other"] + custom_preferences

if final_preferences:
    st.success(f"✅ Selected Preferences: {', '.join(final_preferences)}")

# Generate Itinerary
if st.button("Generate Itinerary"):
    if not destination or not start_location or not final_preferences or trip_days <= 0:
        st.warning("Please fill all fields correctly!")
    else:
        with st.spinner("Planning your dream trip..."):
            prompt = f"""
            Plan a complete round-trip travel itinerary for {people} people.
            Starting from: {start_location}
            Destination: {destination}
            Dates: From {start_date.strftime('%d %B %Y')} to {end_date.strftime('%d %B %Y')} ({trip_days} days)
            Budget range: ₹{min_budget} – ₹{max_budget}
            Preferences: {', '.join(final_preferences)}
            Preferred Transport: {transport_mode}
            Food Type: {food_type}

            Ensure:
            - Add an outbound journey from {start_location} to {destination} and a return trip back
            - Estimate travel duration for each leg (based on mode of transport)
            - Include hotel/accommodation suggestions with approximate prices
            - Suggest 1–3 activities per day
            - Include food options based on the food preference
            - Mention total estimated cost for the trip (roughly)
            - Format clearly with daily breakdown

            Format:
            Day 1: Travel + Activities
            Day 2: ...
            ...
            Final Day: Return trip
            """

            try:
                response = model.generate_content(prompt)
                itinerary = response.text
                st.subheader("🗓️ Your Itinerary:")
                st.markdown(itinerary)

                # PDF download
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                for line in itinerary.split("\n"):
                    try:
                        pdf.cell(200, 10, txt=line.encode("latin-1", "replace").decode("latin-1"), ln=True)
                    except:
                        pass

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    pdf.output(tmp_file.name)
                    with open(tmp_file.name, "rb") as f:
                        st.download_button("📥 Download Itinerary as PDF", f, file_name="itinerary.pdf")

            except Exception as e:
                st.error(f"Error generating itinerary: {e}")