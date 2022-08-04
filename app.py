import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from prediction import get_prediction, encode_value
from PIL import Image


model = joblib.load(r'final_model.joblib')

st.image('Banner.jpg')

features = ['facility_type','energy_star_rating','State_Factor','days_below_20F','building_class','spring_avg_temp','snowdepth_inches',
            'days_above_100F','heating_degree','max_wind_speed','precipitation_inches','fall_avg_temp','floor_area',
            'direction_max_wind_speed','avg_temp','days_below_10F','year_built','summer_min_temp','min_temp','snowfall_inches']

option_facility_type = ['Food_Grocery', 'Warehouse', 'Retail', 'Education', 'Office','Data_Center', 'Commercial', 'Industrial',
                        'Laboratory','Public_Assembly', 'Lodging', 'Health_Care', 'Religious_worship','Parking_Garage', 'Services', 
                        'Unit_Building', 'Multifamily','Public_Safety', 'Mixed_Use']

option_state_factor = ['State_1', 'State_2', 'State_4', 'State_6', 'State_8', 'State_10', 'State_11']

option_building_class = ['Commercial', 'Residential']

def main():
        
    st.subheader("Enter the following scenario:")
    
    #characteristic of the building
    
    facility_type = st.selectbox("Facility type of the building", options=option_facility_type)
    state_factor = st.selectbox("State in which the building is located", options=option_state_factor)
    building_class = st.selectbox("Building Class", options=option_building_class)
    year_built = st.slider("Year in which the building was constructed", 1600, 2016)
    energy_star_rating = st.slider("Energy Star Rating of the building", 0.0, 100.0,0.1)
    floor_area = st.slider("Floor area (square feet) of the building", 943, 6385382)
    
    #environment of the building
    days_below_20F = st.slider("Days with below 20F temperature",0, 93,1)
    days_below_10F = st.slider("Days with below 10F temperature",0, 59,1)
    days_above_100F = st.slider("Days with above 100F temperature",0, 1,1)   
    heating_degree = st.slider("Total months where the daily average temperature falls under 65 Fahreinheit",33,660,1)
    
    spring_avg_temp = st.slider("Average temperature (in Fahreinheit) during spring season at building location",41.5, 75.15,0.1)
    fall_avg_temp = st.slider("Average temperature (in Fahreinheit) during fall season at building location",47.0, 79.0,0.1)
    summer_min_temp = st.slider("Minimum temperature (in Fahreinheit) during summer season at building location",30.0, 68.0,0.1)
    min_temp = st.slider("Minimum temperature (in Fahreinheit) at building location",-19.0, 44.0,0.1)
    avg_temp = st.slider("Average temperature (in Fahreinheit) at building location",44.0, 77.0,0.1)
    
    max_wind_speed = st.slider("Maximum wind speed recorded in the building site",1.0, 23.3,0.1)
    direction_max_wind_speed = st.slider("Direction of the maximum wind speed in the building site",1,360,1)
    
    snowdepth_inches = st.slider("Annual snow depth (inches) at the building site",0, 1292,1)
    snowfall_inches = st.slider("Annual snow fall (inches) at the building site",0.0, 127.0,0.1)
    precipitation_inches = st.slider("Annual precipitation (inches) at the location of the building",0.0, 112.92,0.1)  
           
    submit = st.form_submit_button("Predict")

    if submit:
        facility_type = encode_value(facility_type,option_facility_type)
        state_factor = encode_value(state_factor,option_state_factor)
        building_class = encode_value(building_class,option_building_class)

        data = np.array([accident_cause,casualties,day_of_week,day_time,accident_area,
                         juntion_type,lighting_cond,vehicles_involved,vehicle_type,driving_experience]).reshape(1,-1)
        #st.write(data)
        pred = get_prediction(data=data, model=model)
        st.write(f"🔌 The predicted site energy of the building is:  {pred[0]} 🔌")
           

if __name__ == '__main__':
    main()