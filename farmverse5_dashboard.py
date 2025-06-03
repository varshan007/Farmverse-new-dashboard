import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import random
import uuid
import hashlib # Import hashlib for SHA256 simulation
import xgboost as xgb # Import XGBoost
from PIL import Image # For image upload display

# --- IMPORTANT: st.set_page_config() MUST BE THE FIRST STREAMLIT COMMAND ---
st.set_page_config(layout="wide", page_title="FarmVerse Dashboard",
                   initial_sidebar_state="expanded")

#--- Custom CSS for a Catchy Design ---
st.markdown("""
<style>
.reportview-container {
    background: #f0f2f6; /* Light gray background */
}
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.1rem; /* Larger tab labels */
}
h1 {
    color: #0e8c4f; /* Green for main title */
    text-align: center;
    font-size: 2.8em;
}
h3 {
    color: #2e7d32; /* Slightly darker green for subheaders */
    font-size: 1.6em;
}
.stButton>button {
    background-color: #4CAF50; /* Green buttons */
    color: white;
    border-radius: 5px;
    border: none;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    transition-duration: 0.4s;
}
.stButton>button:hover {
    background-color: #45a049; /* Darker green on hover */
    color: white;
}
.stAlert {
    border-left: 6px solid;
    border-radius: 5px;
}
.stAlert.success {
    border-color: #28a745;
    background-color: #d4edda;
}
.stAlert.info {
    border-color: #17a2b8;
    background-color: #d1ecf1;
}
.stAlert.warning {
    border-color: #ffc107;
    background-color: #fff3cd;
}
.stAlert.error {
    border-color: #dc3545;
    background-color: #f8d7da;
}
/* Custom container styling */
.stContainer {
    border: 1px solid #e6e6e6;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
    background-color: white;
}
.metric-card {
    border: 1px solid #dcdcdc;
    border-radius: 8px;
    padding: 15px;
    text-align: center;
    background-color: #f9f9f9;
    box-shadow: 1px 1px 4px rgba(0,0,0,0.05);
}
.metric-label {
    font-size: 0.9em;
    color: #555;
}
.metric-value {
    font-size: 1.8em;
    font-weight: bold;
    color: #0e8c4f;
}
.quality-param-pass {
    color: green;
    font-weight: bold;
}
.quality-param-fail {
    color: red;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# --- Quality Standards Definition ---
QUALITY_STANDARDS = {
    "Organic Wheat": {
        "Grain Moisture Content": {"ideal_min": 10.0, "ideal_max": 12.0, "unit": "%", "source": "iot", "iot_key": "grain_moisture_content", "note": "Direct grain moisture reading."},
        "Foreign Matter": {"max_allowed": 0.75, "unit": "%", "source": "manual_input"},
        "Broken Grains": {"max_allowed": 3.0, "unit": "%", "source": "manual_input"},
        "Damaged Grains": {"max_allowed": 2.0, "unit": "%", "source": "manual_input"},
        "Insect Infestation": {"max_allowed": 0.0, "unit": " (0=No, 1=Yes)", "source": "manual_input_select", "options": [0, 1], "note": "0 indicates no infestation."},
        "Pesticide Residues": {"max_allowed": 0.01, "unit": "ppm", "source": "lab_report_placeholder", "note": "Requires lab report upload (not implemented)."}
    },
    "Basmati Rice": {
        "Grain Moisture Content": {"ideal_min": 10.0, "ideal_max": 12.0, "unit": "%", "source": "iot", "iot_key": "grain_moisture_content", "note": "Direct grain moisture reading."},
        "Foreign Matter": {"max_allowed": 0.75, "unit": "%", "source": "manual_input"},
        "Broken Grains": {"max_allowed": 5.0, "unit": "%", "source": "manual_input"}, 
        "Damaged Grains": {"max_allowed": 2.0, "unit": "%", "source": "manual_input"},
    },
    "Mango": {
        "Uniform Size & Shape": {"min_ideal": 85.0, "unit": "%", "source": "manual_input", "note": "Visual check percentage."},
        "Blemishes/Cuts": {"max_allowed_area": 5.0, "unit": "% area", "source": "manual_input", "note": "Visual check of surface area."},
        "Brix (Sugar Level)": {"min_ideal": 12.0, "unit": "%", "source": "manual_input"},
        "Moisture Loss (Transit)": {"max_allowed": 5.0, "unit": "%", "source": "manual_input"}, # This is post-harvest, so manual
        "Ripeness": {"ideal_min": 80.0, "ideal_max": 90.0, "unit": "%", "source": "manual_input", "note": "Visual or tactile check."},
        "Bruised/Rotten Units": {"max_allowed": 1.0, "unit": "%", "source": "manual_input"}
    },
    "Wheat": { 
        "Grain Moisture Content": {"ideal_min": 10.0, "ideal_max": 12.0, "unit": "%", "source": "iot", "iot_key": "grain_moisture_content", "note": "Direct grain moisture reading."},
        "Foreign Matter": {"max_allowed": 0.75, "unit": "%", "source": "manual_input"},
    },
     "Corn": {
        "Grain Moisture Content": {"ideal_min": 10.0, "ideal_max": 14.0, "unit": "%", "source": "iot", "iot_key": "grain_moisture_content", "note": "Direct grain moisture reading."},
        "Foreign Matter": {"max_allowed": 1.0, "unit": "%", "source": "manual_input"},
    },
    "Organic Corn": {
        "Grain Moisture Content": {"ideal_min": 10.0, "ideal_max": 14.0, "unit": "%", "source": "iot", "iot_key": "grain_moisture_content", "note": "Direct grain moisture reading."},
        "Foreign Matter": {"max_allowed": 0.75, "unit": "%", "source": "manual_input"},
        "Pesticide Residues": {"max_allowed": 0.01, "unit": "ppm", "source": "lab_report_placeholder", "note": "Requires lab report upload (not implemented)."}
    },
    "Turmeric": { # Example from PDF
        "Moisture Content": {"max_allowed": 10.0, "unit": "%", "source": "manual_input", "note": "Prevents mold."},
        "Foreign Matter": {"max_allowed": 1.0, "unit": "%", "source": "manual_input"},
        "Curcumin Content": {"min_ideal": 2.5, "unit": "%", "source": "manual_input", "note": "Indicates medicinal value."}
    },
    "Groundnut": { # Example from PDF
        "Moisture Content": {"max_allowed": 8.0, "unit": "%", "source": "manual_input", "note": "Avoids damage."},
        "Oil Content": {"min_ideal": 44.0, "unit": "%", "source": "manual_input"},
        "Aflatoxins": {"max_allowed": 20.0, "unit": "ppb", "source": "lab_report_placeholder", "note": "For safety."}
    }
}

# Helper to get a unique key for session state for manual inputs
def get_quality_input_key(product_name, param_name):
    product_key = "".join(filter(str.isalnum, product_name))
    param_key = "".join(filter(str.isalnum, param_name))
    return f"quality_input_{product_key}_{param_key}"

# --- Reward Constants ---
FVT_REWARD_QUALITY_VALIDATION = 10.0 # FVT
FVT_REWARD_CARBON_VERIFICATION = 20.0 # FVT
CARBON_MULTIPLIER_BONUS = 0.05 # 5% bonus
IOT_STREAK_THRESHOLD_POINTS = 5 
LOYALTY_POINTS_REWARD = 50 # Example points
SUSTAINABLE_TASK_COMPLETION_FVT = 5.0 # FVT for 100% sustainable tasks

# --- Badge Definitions ---
BADGE_DEFINITIONS = {
    "Eco Warrior": {"criteria_text": "Offset >100 kg CO₂ (Earned > 4 Carbon Credits)", "threshold_cc_earned": 4.0, "icon": "🌳"}, # Assuming 1 CC ~ 25kg CO2 for demo
    "Quality King": {"criteria_text": "5 successful commodity validations", "threshold_validations": 5, "icon": "👑"},
    "Contract Master": {"criteria_text": "10 flawless smart contracts completed", "threshold_contracts": 10, "icon": "📜"},
    "IoT Consistent": {"criteria_text": f"Consistent IoT data updates ({IOT_STREAK_THRESHOLD_POINTS}-day streak)", "threshold_streak": IOT_STREAK_THRESHOLD_POINTS, "icon": "📡"}
}


#--- 0. Al Model Training & Loading (Cached to run once)
@st.cache_resource
def train_advisory_model():
    print("--- Training Advisory Al Model ---")
    num_samples = 1000
    np.random.seed(42)
    data = {
        'soil_moisture': np.random.uniform(0, 100, num_samples),
        'avg_temp': np.random.uniform(-5, 45, num_samples), 
        'humidity': np.random.uniform(0, 100, num_samples), 
        'pH': np.random.uniform(4.0, 9.0, num_samples), 
        'nitrogen': np.random.uniform(0, 200, num_samples), 
        'phosphorus': np.random.uniform(0, 150, num_samples), 
        'potassium': np.random.uniform(0, 300, num_samples), 
        'rainfall': np.random.uniform(0, 200, num_samples), 
        'crop_yield_t_per_acre': np.random.uniform(0, 10, num_samples), 
        'soil_organic_matter': np.random.uniform(0, 10, num_samples), 
        'grain_moisture_content': np.random.uniform(8, 18, num_samples) # Added for grain
    }
    df = pd.DataFrame(data)
    df['irrigation_needed'] = (df['soil_moisture'] < 40)*1 + (df['rainfall'] < 50) * 0.5 + \
                               np.random.normal(0, 0.1, num_samples) 
    df['pest_risk'] = (df['humidity'] > 70)* 0.3 + (df['avg_temp'] > 30) * 0.2 + np.random.normal(0, 0.1, num_samples) 
    df['fertilization_recommendation'] = (df['nitrogen'] < 100)*1+ (df['phosphorus'] <50) * 0.5 + \
                                         (df['potassium'] < 150)*0.5 + np.random.normal(0, 0.1, num_samples) 
    df['yield_prediction'] = df['crop_yield_t_per_acre']* (1 + np.random.normal(0, 0.05, num_samples)) 
    df['energy_savings_potential'] = (df['avg_temp'] <20)*0.1 + (df['humidity'] < 50) * 0.1 + \
                                     np.random.normal(0, 0.05, num_samples) 
    df['land_cover_health'] = (df['soil_organic_matter'] > 5) * 0.2 + (df['rainfall'] > 100)*0.1 + \
                              np.random.normal(0, 0.05, num_samples) 
    features = [
        'soil_moisture', 'avg_temp', 'humidity', 'pH', 'nitrogen', 'phosphorus', 'potassium',
        'rainfall', 'crop_yield_t_per_acre', 'soil_organic_matter', 'grain_moisture_content' # Added
    ]
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    models = {}
    for target in ['irrigation_needed', 'pest_risk', 'fertilization_recommendation', 'yield_prediction',
                   'energy_savings_potential', 'land_cover_health']:
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_scaled, df[target])
        models[target] = model
    return scaler, models, features

@st.cache_resource
def train_carbon_credit_model():
    print("--- Training Carbon Credit Al Model (XGBoost) ---")
    num_samples = 1000
    np.random.seed(42)
    data = {
        'soil_moisture': np.random.uniform(0, 100, num_samples), 
        'avg_temp': np.random.uniform(-5, 45, num_samples), 
        'humidity': np.random.uniform(0, 100, num_samples), 
        'pH': np.random.uniform(4.0, 9.0, num_samples), 
        'nitrogen': np.random.uniform(0, 200, num_samples), 
        'phosphorus': np.random.uniform(0, 150, num_samples), 
        'potassium': np.random.uniform(0, 300, num_samples), 
        'rainfall': np.random.uniform(0, 200, num_samples), 
        'soil_organic_matter': np.random.uniform(0, 10, num_samples), 
        'crop_yield_t_per_acre': np.random.uniform(0, 10, num_samples), 
        'tillage_practice': np.random.randint(0, 2, num_samples), 
        'organic_inputs_used': np.random.randint(0, 2, num_samples), 
        'cover_crops_used': np.random.randint(0, 2, num_samples), 
        'efficient_water_usage_done': np.random.randint(0, 2, num_samples), 
    }
    df = pd.DataFrame(data)
    df['carbon_credits_potential'] = (
        df['soil_moisture'] * 0.05
        + df['avg_temp'] * 0.01
        + df['humidity'] * 0.01
        + df['pH'] * 0.5
        + df['nitrogen'] * 0.02 + df['phosphorus'] * 0.01 + df['potassium'] * 0.01
        + df['rainfall'] * 0.01
        + df['soil_organic_matter'] * 10.0 
        + df['crop_yield_t_per_acre'] * 0.2
        + df['tillage_practice'] * 15.0 
        + df['organic_inputs_used'] * 8.0 
        + df['cover_crops_used'] * 7.0 
        + df['efficient_water_usage_done'] * 3.0
        + np.random.normal(0, 2, num_samples)
    )
    df['carbon_credits_potential'] = np.maximum(0, df['carbon_credits_potential'])
    features = [
        'soil_moisture', 'avg_temp', 'humidity', 'pH', 'nitrogen', 'phosphorus', 'potassium',
        'rainfall', 'soil_organic_matter', 'crop_yield_t_per_acre', 'tillage_practice',
        'organic_inputs_used', 'cover_crops_used', 'efficient_water_usage_done'
    ]
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = xgb.XGBRegressor(n_estimators=100, random_state=42, tree_method='hist',
                             enable_categorical=False) 
    model.fit(X_scaled, df['carbon_credits_potential'])
    return scaler, model, features

@st.cache_resource
def train_credit_score_model():
    print("--- Training Credit Score Al Model (XGBoost) ---")
    num_samples = 1000
    np.random.seed(42)
    data = {
        'iot_data_consistency': np.random.uniform(50, 100, num_samples),
        'marketplace_performance': np.random.uniform(50, 100, num_samples),
        'smart_contract_fulfillment': np.random.uniform(50, 100, num_samples),
        'carbon_earnings_impact': np.random.uniform(50, 100, num_samples),
        'resource_sharing_activity': np.random.uniform(0, 100, num_samples), 
        'advisory_adherence': np.random.uniform(50, 100, num_samples),
        'yield_consistency': np.random.uniform(50, 100, num_samples),
    }
    df = pd.DataFrame(data)
    df['credit_score'] = (
        df['iot_data_consistency'] * 0.20 +
        df['marketplace_performance'] * 0.25 +
        df['smart_contract_fulfillment'] * 0.20 +
        df['carbon_earnings_impact'] * 0.15 +
        df['resource_sharing_activity'] * 0.10 +
        df['advisory_adherence'] * 0.05 +
        df['yield_consistency'] * 0.05 +
        np.random.normal(0, 5, num_samples) 
    )
    df['credit_score'] = np.clip(df['credit_score'], 0, 100) 
    features = list(data.keys())
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = xgb.XGBRegressor(n_estimators=100, random_state=42, tree_method='hist',
                             enable_categorical=False)
    model.fit(X_scaled, df['credit_score'])
    return scaler, model, features

#--- Helper Functions
def generate_hash(data_string):
    return hashlib.sha256(data_string.encode('utf-8')).hexdigest()

def add_to_ledger(transaction_desc):
    prev_hash = "0"*64 
    if st.session_state.blockchain_ledger:
        prev_hash = st.session_state.blockchain_ledger[-1]["current_hash"]
    block_data_content = {
        "block_id": st.session_state.next_block_id,
        "timestamp": datetime.now().isoformat(),
        "transaction": transaction_desc
    }
    current_hash = generate_hash(str(block_data_content)) 
    st.session_state.blockchain_ledger.append({
        "block_id": st.session_state.next_block_id,
        "timestamp": datetime.now().isoformat(),
        "transaction": transaction_desc,
        "prev_hash": prev_hash,
        "current_hash": current_hash,
        "data": transaction_desc 
    })
    st.session_state.next_block_id += 1
    if len(st.session_state.blockchain_ledger) > 20: 
        st.session_state.blockchain_ledger.pop(0)

def display_message(placeholder, type, text):
    if type == "success": placeholder.success(text)
    elif type == "warning": placeholder.warning(text)
    elif type == "error": placeholder.error(text)
    else: placeholder.info(text)

#--- Al Model Prediction Functions
def run_advisory_predictions():
    input_data = {
        'soil_moisture': st.session_state.iot_data['soil_moisture'],
        'avg_temp': st.session_state.iot_data['avg_temp'],
        'humidity': st.session_state.iot_data['humidity'],
        'pH': st.session_state.iot_data['soil_pH'],
        'nitrogen': st.session_state.iot_data['soil_nitrogen'],
        'phosphorus': st.session_state.iot_data['soil_phosphorus'],
        'potassium': st.session_state.iot_data['soil_potassium'],
        'rainfall': st.session_state.iot_data['rainfall'],
        'crop_yield_t_per_acre': st.session_state.iot_data['crop_yield_t_per_acre'],
        'soil_organic_matter': st.session_state.iot_data['soil_organic_matter'],
        'grain_moisture_content': st.session_state.iot_data['grain_moisture_content'] # Added
    }
    input_df = pd.DataFrame([input_data], columns=st.session_state.advisory_features)
    input_scaled = st.session_state.advisory_scaler.transform(input_df)
    advisory = {
        "irrigation_needed": "No immediate irrigation needed.",
        "pest_risk": "Low pest risk. Monitor for aphids.",
        "fertilization_recommendation": "Optimal NPK levels. No immediate fertilization needed.",
        "yield_prediction": "Estimated yield: N/A T/Acre.",
        "energy_savings_potential": "Potential N/A% energy savings.",
        "land_cover_health": "N/A land cover health.",
        "pesticide_application": "No pesticide application recommended." 
    }
    for target, model in st.session_state.advisory_models.items():
        prediction = model.predict(input_scaled)[0]
        if target == 'irrigation_needed':
            advisory[target] = "High irrigation needed." if prediction > 1.5 else ("Moderate irrigation needed." if prediction > 0.5 else "No immediate irrigation needed.")
        elif target == 'pest_risk':
            advisory[target] = "High pest risk! Take immediate action." if prediction > 0.8 else ("Moderate pest risk. Monitor closely." if prediction > 0.4 else "Low pest risk. Monitor for aphids.")
        elif target == 'fertilization_recommendation':
            advisory[target] = "High fertilization needed (N, P, K)." if prediction > 1.5 else ("Moderate fertilization needed." if prediction > 0.5 else "Optimal NPK levels. No immediate fertilization needed.")
        elif target == 'yield_prediction':
            advisory[target] = f"Estimated yield: {prediction:.2f} T/Acre."
        elif target == 'energy_savings_potential':
            advisory[target] = f"Potential {prediction * 100:.1f}% energy savings with optimized pump usage."
        elif target == 'land_cover_health':
            advisory[target] = "Land cover health is excellent." if prediction > 0.2 else ("Good land cover health." if prediction > 0.1 else "Improve land cover for better soil health.")
    st.session_state.advisory_output = advisory
    current_date = date.today()
    if not st.session_state.advisory_history or st.session_state.advisory_history[-1][0] != current_date:
        st.session_state.advisory_history.append((current_date, st.session_state.advisory_output.copy()))
    if len(st.session_state.advisory_history) > 30:
        st.session_state.advisory_history.pop(0)

def get_carbon_credit_improvement_suggestions(iot_data, tasks):
    suggestions = []
    if iot_data['soil_moisture'] < 10: suggestions.append("Increase soil moisture levels. Target: 10-60%.")
    elif iot_data['soil_moisture'] > 60: suggestions.append("Reduce soil moisture. Target: 10-60%.")
    if iot_data['soil_nitrogen'] < 20: suggestions.append("Apply nitrogen-rich organic fertilizers. Optimal N: 20-100 mg/kg.")
    if iot_data['soil_phosphorus'] < 15: suggestions.append("Supplement with phosphorus. Optimal P: 15-100 mg/kg.")
    if iot_data['soil_potassium'] < 50: suggestions.append("Increase potassium levels. Optimal K: 50-250 mg/kg.")
    if not (5.5 <= iot_data['soil_pH'] <= 7.5): suggestions.append(f"Adjust soil pH from {iot_data['soil_pH']:.1f} to 5.5-7.5.")
    if iot_data['soil_organic_matter'] < 5.0: suggestions.append(f"Increase soil organic matter (current {iot_data['soil_organic_matter']:.1f}%).")
    if iot_data['tillage_practice'] == 0: suggestions.append("Consider no-till or reduced tillage.")
    
    task_params_done = {task.get('param') for task in tasks if task.get('sustainable_practice') and task['done']}
    if 'organic_inputs_used' not in task_params_done: suggestions.append("Incorporate more organic inputs.")
    if 'cover_crops_used' not in task_params_done: suggestions.append("Plant cover crops.")
    if 'efficient_water_usage_done' not in task_params_done: suggestions.append("Implement efficient water usage.")
    
    if iot_data['humidity'] < 40: suggestions.append("Maintain good microclimate humidity for soil carbon.")
    if not suggestions: suggestions.append("Excellent practices for carbon sequestration!")
    return suggestions

def calculate_carbon_credits():
    iot = st.session_state.iot_data
    carbon_input_data = {
        'soil_moisture': iot['soil_moisture'], 'avg_temp': iot['avg_temp'], 'humidity': iot['humidity'],
        'pH': iot['soil_pH'], 'nitrogen': iot['soil_nitrogen'], 'phosphorus': iot['soil_phosphorus'],
        'potassium': iot['soil_potassium'], 'rainfall': iot['rainfall'], 
        'soil_organic_matter': iot['soil_organic_matter'], 
        'crop_yield_t_per_acre': iot['crop_yield_t_per_acre'],
        'tillage_practice': iot['tillage_practice'],
        'organic_inputs_used': 0, 'cover_crops_used': 0, 'efficient_water_usage_done': 0,
    }
    for task in st.session_state.todays_tasks:
        if task['done'] and task.get('sustainable_practice') and task.get('param') in carbon_input_data:
            carbon_input_data[task['param']] = 1
    
    input_df = pd.DataFrame([carbon_input_data], columns=st.session_state.carbon_features)
    input_scaled = st.session_state.carbon_scaler.transform(input_df)
    potential_credits = st.session_state.carbon_model.predict(input_scaled)[0]

    # Carbon Credit Multiplier Logic
    bonus_multiplier = 1.0
    conditions_met_for_bonus = True
    if not (iot['soil_organic_matter'] > 2.5): conditions_met_for_bonus = False
    if not (6.5 <= iot['soil_pH'] <= 7.2): conditions_met_for_bonus = False
    # Using iot_update_streak from farmer_profile
    if st.session_state.farmer_profile.get('iot_update_streak', 0) < BADGE_DEFINITIONS["IoT Consistent"]["threshold_streak"]:
        conditions_met_for_bonus = False
    
    if conditions_met_for_bonus:
        bonus_multiplier += CARBON_MULTIPLIER_BONUS
        st.toast(f"🌿 Carbon credit bonus of {CARBON_MULTIPLIER_BONUS*100:.0f}% applied for excellent sustainable practices!", icon="🎉")
    
    potential_credits *= bonus_multiplier
    st.session_state.carbon_credits['pending_mint'] = max(0, float(potential_credits))

    feature_importances = st.session_state.carbon_model.feature_importances_
    features_df = pd.DataFrame({'Feature': st.session_state.carbon_features, 'Importance': feature_importances})
    st.session_state.carbon_credit_feature_impact = features_df.sort_values(by='Importance', ascending=False)
    st.session_state.carbon_credit_suggestions = get_carbon_credit_improvement_suggestions(iot, st.session_state.todays_tasks)

def calculate_credit_score():
    factors = st.session_state.credit_score_factors
    iot = st.session_state.iot_data
    sc_metrics = st.session_state.smart_contract
    cc_metrics = st.session_state.carbon_credits
    
    sm_score = 100 - abs(iot['soil_moisture'] - 50) * 0.8
    temp_score = 100 - abs(iot['avg_temp'] - 28) * 1.5
    hum_score = 100 - abs(iot['humidity'] - 60) * 0.8
    factors['iot_data_consistency'] = np.clip(np.mean([sm_score, temp_score, hum_score]), 0, 100)

    max_sales = 10000.0
    factors['marketplace_performance'] = np.clip((st.session_state.marketplace_metrics['sales_volume'] / max_sales) * 100, 0, 100) if max_sales > 0 else 0.0
    
    total_contracts = sc_metrics['completed_successful'] + sc_metrics['in_dispute']
    factors['smart_contract_fulfillment'] = (sc_metrics['completed_successful'] / total_contracts) * 100 if total_contracts > 0 else 98.0

    max_carbon = 100.0
    factors['carbon_earnings_impact'] = np.clip((float(cc_metrics.get('earned_total',0.0)) / max_carbon) * 100, 0, 100) if max_carbon > 0 else 0.0
    
    factors['resource_sharing_activity'] = st.session_state.resource_sharing_activity_score
    
    sustainable_tasks = [t for t in st.session_state.todays_tasks if t.get('sustainable_practice')]
    completed_sustainable = sum(1 for t in sustainable_tasks if t['done'])
    factors['advisory_adherence'] = (completed_sustainable / len(sustainable_tasks)) * 100 if sustainable_tasks else 88.0

    deviation = abs(iot['crop_yield_t_per_acre'] - 3.0)
    factors['yield_consistency'] = np.clip(100 - (deviation * 50), 0, 100)

    input_data = {k: float(v) for k, v in factors.items()}
    input_df = pd.DataFrame([input_data], columns=st.session_state.credit_features)
    input_scaled = st.session_state.credit_scaler.transform(input_df)
    base_score = st.session_state.credit_model.predict(input_scaled)[0]
    st.session_state.overall_credit_score = np.clip(float(base_score) * 8, 0, 800)

    suggestions = []
    if factors['iot_data_consistency'] < 80: suggestions.append("Improve IoT data consistency.")
    if factors['marketplace_performance'] < 75: suggestions.append("Increase marketplace sales.")
    if factors['smart_contract_fulfillment'] < 80: suggestions.append("Focus on successful smart contracts.")
    if factors['carbon_earnings_impact'] < 70: suggestions.append("Earn more carbon credits.")
    if factors['resource_sharing_activity'] < 60: suggestions.append("Engage more in resource sharing.")
    if factors['advisory_adherence'] < 80: suggestions.append("Follow AI advisories closely.")
    if factors['yield_consistency'] < 70: suggestions.append("Improve yield consistency.")
    st.session_state.credit_improvement_suggestions = suggestions if suggestions else ["Your credit profile is strong!"]

def update_market_prices():
    today = date.today()
    for produce_type in st.session_state.marketplace_prices:
        st.session_state.marketplace_prices[produce_type] = [item for item in st.session_state.marketplace_prices[produce_type] if today - item[0] < timedelta(days=7)]
        if st.session_state.marketplace_prices[produce_type]:
            last_date, last_val = st.session_state.marketplace_prices[produce_type][-1]
            new_price = last_val * (1 + random.uniform(-0.02, 0.02))
            new_price = max(100.0, new_price)
            if last_date == today: st.session_state.marketplace_prices[produce_type][-1] = (today, new_price)
            else: st.session_state.marketplace_prices[produce_type].append((today, new_price))
        else:
            base = 200.0; 
            if produce_type == "Rice": base = 300.0
            elif produce_type == "Corn": base = 150.0
            st.session_state.marketplace_prices[produce_type] = [(today - timedelta(days=i), random.uniform(base*0.9, base*1.1)) for i in range(6, -1, -1)]

def award_badge_if_eligible(badge_name, condition_met, farmer_profile_state):
    if condition_met and badge_name not in farmer_profile_state.get('badges', []):
        farmer_profile_state.setdefault('badges', []).append(badge_name)
        badge_info = BADGE_DEFINITIONS.get(badge_name, {})
        st.toast(f"{badge_info.get('icon','🏅')} New Badge Unlocked: {badge_name}!", icon="🎉")
        add_to_ledger(f"Badge Awarded: {badge_name} to {farmer_profile_state['name']}")

#--- 1. Session State Initialization
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.show_profile_edit_form = False 

    st.session_state.advisory_scaler, st.session_state.advisory_models, st.session_state.advisory_features = train_advisory_model()
    st.session_state.carbon_scaler, st.session_state.carbon_model, st.session_state.carbon_features = train_carbon_credit_model()
    st.session_state.credit_scaler, st.session_state.credit_model, st.session_state.credit_features = train_credit_score_model()

    st.session_state.farmer_profile = {
        "name": "Rajesh Kumar", "location": "Chennai, Tamil Nadu", "land_holding_size": 5.0, 
        "crop_type": "Wheat", "tillage_date": date(2025, 5, 1), "sowing_date": date(2025, 5, 10),
        "farmverse_digital_id": "FV-FARM-007", 
        "other_info": "Member of local cooperative, interested in organic farming techniques.",
        "badges": [], "loyalty_points": 0, "iot_update_streak": 0, # For rewards
        "co2_offset_total": 0.0, "successful_validations_count": 0, "flawless_contracts_count": 0 # For badges
    }
    st.session_state.iot_data = {
        "soil_moisture": 50.0, "soil_nitrogen": 80.0, "soil_phosphorus": 45.0, 
        "soil_potassium": 180.0, "soil_pH": 6.5, "avg_temp": 28.0, "humidity": 70.0, 
        "rainfall": 5.0, "soil_organic_matter": 4.5, "crop_yield_t_per_acre": 3.0, 
        "tillage_practice": 1, "weather_forecast_temp": 30.0, 
        "weather_forecast_rain": 10.0, "weather_forecast_humidity": 75.0,
        "grain_moisture_content": 12.5 # New IoT data point for grain
    }
    st.session_state.advisory_output = {"yield_prediction": "Estimated yield: 3.2 T/Acre."} 
    st.session_state.advisory_history = [] 
    st.session_state.todays_tasks = [
        {"task": "Check soil moisture in Sector A", "done": False, "sustainable_practice": False},
        {"task": "Apply organic compost to Crop 1", "done": False, "sustainable_practice": True, "param": "organic_inputs_used"},
        {"task": "Inspect for early signs of rust disease", "done": False, "sustainable_practice": False},
        {"task": "Record daily grain moisture content", "done": False, "sustainable_practice": False} # Example new task
    ]
    st.session_state.sustainable_reward_given_today = False 

    st.session_state.marketplace_prices = {
        "Wheat": [(date.today() - timedelta(days=i), random.uniform(190, 210)) for i in range(6,-1,-1)],
        "Rice": [(date.today() - timedelta(days=i), random.uniform(290, 310)) for i in range(6,-1,-1)],
        "Corn": [(date.today() - timedelta(days=i), random.uniform(140, 160)) for i in range(6,-1,-1)],
    }
    st.session_state.marketplace_listings = [
        {"id": "ML001", "produce": "Organic Wheat", "quantity": 10, "unit": "tons", "price_per_unit": 220.00, "ready_by": date(2025, 7, 15), "status": "Available"},
        {"id": "ML002", "produce": "Basmati Rice", "quantity": 5, "unit": "tons", "price_per_unit": 330.00, "ready_by": date(2025, 6, 30), "status": "Available"}
    ]
    st.session_state.marketplace_metrics = {"sales_volume": 0.0, "transactions_count": 0}

    st.session_state.next_block_id = 0 
    st.session_state.blockchain_ledger = []
    add_to_ledger("Genesis Block: FarmVerse Init") 
    add_to_ledger(f"Farmer {st.session_state.farmer_profile['name']} Onboarding")


    st.session_state.wallets = {"farmer_wallet_balance": 1000.00, "buyer_wallet_balance": 5000.00, "escrow_wallet_balance": 0.00, "fvt_balance": 0.0}
    st.session_state.smart_contract = {
        "status": "Idle", "progress": 0, "current_contract_details": {}, "active_contracts": 0,
        "completed_successful": 0, "in_dispute": 0, "last_payment_status": "N/A",
        "show_logistics_form": False, "logistics_details": {}, "pending_delivery": 0, 
        "show_iot_quality_check": False, "quality_inputs": {}, "current_quality_report": None
    }
    st.session_state.carbon_credits = {
        "earned_total": 0.0, "sold_total": 0.0, "pending_mint": 0.0, "market_price_per_credit": 25.00, 
        "verification_status": "Pending Calculation", "verified_credits": 0.0, "verification_org": "Verra" 
    }
    st.session_state.carbon_credit_feature_impact = pd.DataFrame() 
    st.session_state.carbon_credit_suggestions = [] 
    st.session_state.credit_improvement_suggestions = [] 

    st.session_state.dao = {
        "members": 1500, 
        "proposals": [
            {"id": "DAO-001", "title": "Increase Carbon Credit Payout by 5%", "status": "Voting", "votes_for": 250, "votes_against": 100},
            {"id": "DAO-002", "title": "Fund New Soil Health Research", "status": "Approved", "votes_for": 400, "votes_against": 50},
        ]
    }
    st.session_state.credit_score_factors = {
        "iot_data_consistency": 85.0, "marketplace_performance": 92.0, 
        "smart_contract_fulfillment": 98.0, "carbon_earnings_impact": 90.0, 
        "resource_sharing_activity": 70.0, "advisory_adherence": 88.0, "yield_consistency": 80.0, 
    }
    st.session_state.resource_sharing_activity_score = 70.0 
    st.session_state.overall_credit_score = 0.0 
    st.session_state.financial_linkage_partners = ["ABC Bank", "Rural MFI", "AgriFin Corp"]
    st.session_state.credit_profile_shared_status = False
    st.session_state.available_loans = [
        {"id": "LOAN-001", "name": "Crop Cultivation Loan", "interest_rate": "7.5% p.a.", "max_amount": "$50,000", "eligibility_score": 500, "status": "Available", "application_status": "N/A"},
        {"id": "LOAN-002", "name": "Farm Equipment Loan", "interest_rate": "8.0% p.a.", "max_amount": "$100,000", "eligibility_score": 600, "status": "Available", "application_status": "N/A"},
    ]
    st.session_state.uploaded_quality_image = None 

if st.session_state.initialized:
    run_advisory_predictions()
    calculate_carbon_credits()
    calculate_credit_score() 

#--- Streamlit UI ---
st.title("🌱 FarmVerse: Reimagining Farming for a Connected World")
st.markdown("### Your Team: Varshan PA & Teammates, ICAR-Indian Agricultural Research Institute, New Delhi")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "👨‍🌾 Farmer Profile", "📊 IoT Data & AI Advisory", "🌐 Marketplace & Contracts",
    "🌿 Carbon Credits", "💰 AI Credit Score & Financials", "📈 Analytics & Reports"
])

with tab1:
    st.header("👨‍🌾 Farmer Profile & Onboarding")
    profile_msg_placeholder = st.empty()
    
    with st.container(border=True):
        st.subheader("🏆 Achievements & Rewards")
        col_badges_tab1, col_points_fvt_tab1 = st.columns(2)
        with col_badges_tab1:
            st.write("**Badges Earned:**")
            if st.session_state.farmer_profile.get('badges'):
                for badge_name in st.session_state.farmer_profile['badges']:
                    badge_info = BADGE_DEFINITIONS.get(badge_name, {})
                    st.markdown(f"- {badge_info.get('icon', '🏅')} **{badge_name}**") 
            else:
                st.caption("No badges earned yet. Keep up the good work!")
        with col_points_fvt_tab1:
            st.metric("Loyalty Points", st.session_state.farmer_profile.get('loyalty_points', 0))
            st.metric("FarmVerse Tokens (FVT)", f"{st.session_state.wallets.get('fvt_balance', 0.0):.2f} FVT")
            st.caption(f"IoT Update Streak: {st.session_state.farmer_profile.get('iot_update_streak',0)}")


    with st.container(border=True):
        st.markdown("### Your Farm Details")
        if st.session_state.show_profile_edit_form:
            with st.form("farmer_profile_form_tab1"): 
                col1, col2 = st.columns(2)
                with col1:
                    st.text_input("Farmer Name", value=st.session_state.farmer_profile["name"], key="profile_name_tab1")
                    st.text_input("Location", value=st.session_state.farmer_profile["location"], key="profile_location_tab1")
                    st.number_input("Land Holding Size (Acres)", min_value=0.1,
                                    value=st.session_state.farmer_profile["land_holding_size"], step=0.1, key="profile_land_size_tab1")
                    st.text_input("Crop Type", value=st.session_state.farmer_profile["crop_type"],
                                  key="profile_crop_type_tab1")
                with col2:
                    tillage_date_val = st.session_state.farmer_profile["tillage_date"]
                    if tillage_date_val > date.today(): tillage_date_val = date.today()
                    sowing_date_val = st.session_state.farmer_profile["sowing_date"]
                    if sowing_date_val > date.today(): sowing_date_val = date.today()
                    st.date_input("Last Tillage Date", value=tillage_date_val, key="profile_tillage_date_tab1", max_value=date.today())
                    st.date_input("Sowing Date", value=sowing_date_val, key="profile_sowing_date_tab1", max_value=date.today())
                    st.text_input("FarmVerse Digital ID", value=st.session_state.farmer_profile["farmverse_digital_id"],
                                  disabled=True, key="profile_id_tab1")
                    st.text_area("Other Important Information", value=st.session_state.farmer_profile["other_info"],
                                 height=100, key="profile_other_info_tab1")
                submitted = st.form_submit_button("💾 Save Profile")
                if submitted:
                    st.session_state.farmer_profile.update({
                        "name": st.session_state.profile_name_tab1, "location": st.session_state.profile_location_tab1,
                        "land_holding_size": st.session_state.profile_land_size_tab1, "crop_type": st.session_state.profile_crop_type_tab1,
                        "tillage_date": st.session_state.profile_tillage_date_tab1, "sowing_date": st.session_state.profile_sowing_date_tab1,
                        "other_info": st.session_state.profile_other_info_tab1
                    })
                    add_to_ledger(f"Farmer Profile Updated: {st.session_state.farmer_profile['name']}")
                    display_message(profile_msg_placeholder, "success", "Farmer profile updated successfully!")
                    st.session_state.show_profile_edit_form = False
                    st.rerun()
        else:
            fp = st.session_state.farmer_profile
            col_overview_1, col_overview_2 = st.columns(2)
            with col_overview_1:
                st.write(f"**Farmer Name:** {fp['name']}")
                st.write(f"**Location:** {fp['location']}")
                st.write(f"**Land Holding Size:** {fp['land_holding_size']} Acres")
                st.write(f"**Crop Type:** {fp['crop_type']}")
            with col_overview_2:
                st.write(f"**Last Tillage Date:** {fp['tillage_date']}")
                st.write(f"**Sowing Date:** {fp['sowing_date']}")
                st.write(f"**FarmVerse Digital ID:** {fp['farmverse_digital_id']}")
                st.write(f"**Other Info:** {fp['other_info']}")
            if st.button("✏️ Edit Profile", key="edit_profile_btn_tab1"):
                st.session_state.show_profile_edit_form = True
                st.rerun()


with tab2: 
    st.header("📊 IoT Data & AI Advisory")
    advisory_msg_placeholder = st.empty()
    col_iot_input, col_advisory_output = st.columns(2)

    with col_iot_input:
        with st.container(border=True):
            st.subheader("Simulated IoT Sensor Data")
            st.info("Adjust these values to see how Al advisories change!")
            st.slider("Soil Moisture (%)", 0, 100, value=int(st.session_state.iot_data["soil_moisture"]), key="iot_soil_moisture_tab2")
            st.slider("Grain Moisture Content (%)", 5, 25, value=int(st.session_state.iot_data.get("grain_moisture_content", 12)), key="iot_grain_moisture_tab2") 
            st.number_input("Soil Nitrogen (mg/kg)", 0, 200, value=int(st.session_state.iot_data["soil_nitrogen"]), key="iot_soil_nitrogen_tab2")
            st.number_input("Soil Phosphorus (mg/kg)", 0, 150, value=int(st.session_state.iot_data["soil_phosphorus"]), key="iot_soil_phosphorus_tab2")
            st.number_input("Soil Potassium (mg/kg)", 0, 300, value=int(st.session_state.iot_data["soil_potassium"]), key="iot_soil_potassium_tab2")
            st.number_input("Soil pH", 4.0, 9.0, value=float(st.session_state.iot_data["soil_pH"]), step=0.1, key="iot_soil_pH_tab2")
            st.number_input("Average Temperature (°C)", -5.0, 45.0, value=float(st.session_state.iot_data["avg_temp"]), step=0.1, key="iot_avg_temp_tab2")
            st.number_input("Humidity (%)", 0.0, 100.0, value=float(st.session_state.iot_data["humidity"]), step=0.1, key="iot_humidity_tab2")
            st.number_input("Daily Rainfall (mm)", 0.0, 200.0, value=float(st.session_state.iot_data["rainfall"]), step=0.1, key="iot_rainfall_tab2")
            st.number_input("Soil Organic Matter (%)", 0.0, 10.0, value=float(st.session_state.iot_data["soil_organic_matter"]), step=0.1, key="iot_soil_organic_matter_tab2")
            st.number_input("Harvest Yield Estimate (T/Acre)", 0.0, 10.0, value=float(st.session_state.iot_data["crop_yield_t_per_acre"]), step=0.1, key="iot_crop_yield_tab2")
            st.radio("Tillage Practice", ["Conventional", "No-Till"], index=st.session_state.iot_data["tillage_practice"], key="iot_tillage_practice_radio_tab2")
            
            st.markdown("---"); st.subheader("Simulated Weather Forecast")
            st.number_input("Forecast Temperature (°C)", -5.0, 45.0, value=st.session_state.iot_data["weather_forecast_temp"], step=0.1, key="iot_forecast_temp_tab2")
            st.number_input("Forecast Rainfall (mm)", 0.0, 200.0, value=st.session_state.iot_data["weather_forecast_rain"], step=0.1, key="iot_forecast_rain_tab2")
            st.number_input("Forecast Humidity (%)", 0.0, 100.0, value=st.session_state.iot_data["weather_forecast_humidity"], step=0.1, key="iot_forecast_humidity_tab2")

            if st.button("🔄 Update IoT Data & Get Advisory", key="update_iot_btn_tab2"):
                st.session_state.iot_data.update({
                    "soil_moisture": float(st.session_state.iot_soil_moisture_tab2),
                    "grain_moisture_content": float(st.session_state.iot_grain_moisture_tab2), 
                    "soil_nitrogen": float(st.session_state.iot_soil_nitrogen_tab2),
                    "soil_phosphorus": float(st.session_state.iot_soil_phosphorus_tab2),
                    "soil_potassium": float(st.session_state.iot_soil_potassium_tab2),
                    "soil_pH": float(st.session_state.iot_soil_pH_tab2),
                    "avg_temp": float(st.session_state.iot_avg_temp_tab2),
                    "humidity": float(st.session_state.iot_humidity_tab2),
                    "rainfall": float(st.session_state.iot_rainfall_tab2),
                    "soil_organic_matter": float(st.session_state.iot_soil_organic_matter_tab2),
                    "crop_yield_t_per_acre": float(st.session_state.iot_crop_yield_tab2),
                    "tillage_practice": 0 if st.session_state.iot_tillage_practice_radio_tab2 == "Conventional" else 1,
                    "weather_forecast_temp": float(st.session_state.iot_forecast_temp_tab2),
                    "weather_forecast_rain": float(st.session_state.iot_forecast_rain_tab2),
                    "weather_forecast_humidity": float(st.session_state.iot_forecast_humidity_tab2)
                })
                st.session_state.farmer_profile['iot_update_streak'] = st.session_state.farmer_profile.get('iot_update_streak',0) + 1
                if st.session_state.farmer_profile['iot_update_streak'] >= BADGE_DEFINITIONS["IoT Consistent"]["threshold_streak"]:
                     award_badge_if_eligible("IoT Consistent", True, st.session_state.farmer_profile)
                elif st.session_state.farmer_profile['iot_update_streak'] > 0 and st.session_state.farmer_profile['iot_update_streak'] % IOT_STREAK_THRESHOLD_POINTS == 0 : # Reward for intermediate streaks too
                    st.session_state.farmer_profile['loyalty_points'] += LOYALTY_POINTS_REWARD
                    st.toast(f"✨ Loyalty Bonus! +{LOYALTY_POINTS_REWARD} points (Streak: {st.session_state.farmer_profile['iot_update_streak']})!", icon="👍")
                    add_to_ledger(f"Loyalty Points: {LOYALTY_POINTS_REWARD} for IoT streak.")
                
                run_advisory_predictions(); calculate_carbon_credits(); calculate_credit_score()
                add_to_ledger("IoT Data Updated & Advisory Generated")
                display_message(advisory_msg_placeholder, "success", "IoT data updated and Al advisory generated!")
                st.rerun()
    with col_advisory_output:
        with st.container(border=True):
            st.subheader("AI Advisory for Your Farm")
            st.info("Recommendations based on current and forecast data.")
            for key, value in st.session_state.advisory_output.items():
                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            st.markdown("---")
        with st.container(border=True):
            st.subheader("Today's Task Remainder")
            st.info("Mark tasks as done. Completed sustainable tasks will impact your carbon credits!")
            for i, task_item in enumerate(st.session_state.todays_tasks):
                col_task_desc, col_task_check = st.columns([0.8, 0.2])
                with col_task_desc: st.markdown(f"- **{task_item['task']}** {'♻️' if task_item.get('sustainable_practice') else ''}")
                with col_task_check:
                    checked = st.checkbox("Done", value=task_item["done"], key=f"task_checkbox_{i}_tab2")
                    if checked != task_item["done"]:
                        st.session_state.todays_tasks[i]["done"] = checked
                        if not checked and task_item.get('sustainable_practice'): st.session_state.sustainable_reward_given_today = False
                        calculate_carbon_credits(); calculate_credit_score()
                        add_to_ledger(f"Task '{task_item['task']}' marked {'Done' if checked else 'Undone'}")
                        display_message(advisory_msg_placeholder, "info", f"Task updated: '{task_item['task']}'"); st.rerun()
            st.markdown("---")
            with st.container(border=True):
                st.subheader("Sustainable Practices Progress")
                total_sustainable = sum(1 for t in st.session_state.todays_tasks if t.get('sustainable_practice'))
                completed_sustainable = sum(1 for t in st.session_state.todays_tasks if t['done'] and t.get('sustainable_practice'))
                if total_sustainable > 0:
                    sustainability_progress = (completed_sustainable / total_sustainable) * 100
                    st.progress(sustainability_progress / 100, text=f"Sustainable Tasks: {completed_sustainable}/{total_sustainable} ({sustainability_progress:.1f}%)")
                    if sustainability_progress == 100 and not st.session_state.sustainable_reward_given_today:
                        reward_amount = 50.0 
                        st.session_state.wallets['farmer_wallet_balance'] += reward_amount
                        st.session_state.wallets['fvt_balance'] = st.session_state.wallets.get('fvt_balance',0.0) + SUSTAINABLE_TASK_COMPLETION_FVT
                        add_to_ledger(f"Sustainability Reward: ${reward_amount:.2f} & {SUSTAINABLE_TASK_COMPLETION_FVT} FVT.")
                        display_message(advisory_msg_placeholder, "success", f"🥳 Earned ${reward_amount:.2f} & {SUSTAINABLE_TASK_COMPLETION_FVT} FVT for sustainable tasks!")
                        st.session_state.sustainable_reward_given_today = True; st.rerun()
                else: st.info("No sustainable tasks defined for today.")

#--- Tab 3: Blockchain Marketplace & Contracts ---
with tab3:
    st.header("🌐 Blockchain Marketplace & Smart Contracts")
    marketplace_msg_placeholder = st.empty()
    col_marketplace, col_smart_contracts = st.columns(2)

    with col_marketplace: # Marketplace Listings and Price Trends
        with st.container(border=True):
            st.subheader("Live Marketplace Prices (Past Week)")
            if st.session_state.marketplace_prices:
                produce_options = list(st.session_state.marketplace_prices.keys())
                if produce_options:
                    selected_produce_price = st.selectbox("Select Produce for Price Trend", produce_options, key="price_trend_select_tab3_unique")
                    if selected_produce_price and st.session_state.marketplace_prices.get(selected_produce_price):
                        price_data = pd.DataFrame(st.session_state.marketplace_prices[selected_produce_price], columns=["Date", "Price"])
                        price_data["Date"] = pd.to_datetime(price_data["Date"])
                        fig = px.line(price_data, x="Date", y="Price", title=f"{selected_produce_price} Price Trend", labels={"Price": "Price (USD/Unit)"}, template="plotly_white")
                        fig.update_traces(mode='lines+markers')
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No produce types available for price trends yet.")
            if st.button("📊 Simulate Daily Price Fluctuation", key="simulate_price_fluctuation_btn_tab3_unique"):
                update_market_prices(); display_message(marketplace_msg_placeholder, "info", "Market prices simulated."); st.rerun()

        with st.container(border=True):
            st.subheader("Your Available Produce Listings")
            if st.session_state.marketplace_listings: st.dataframe(pd.DataFrame(st.session_state.marketplace_listings), use_container_width=True, hide_index=True)
            else: st.info("No listings currently available.")

        with st.container(border=True):
            st.subheader("➕ Add New Produce Listing")
            with st.form("new_listing_form_tab3_unique"):
                available_produce_for_listing = list(QUALITY_STANDARDS.keys())
                if not available_produce_for_listing:
                    st.warning("No produce types with quality standards defined. Please add them to QUALITY_STANDARDS.")
                    st.form_submit_button("List Produce", disabled=True)
                else:
                    new_produce = st.selectbox("Produce Name", available_produce_for_listing, key="new_produce_select_tab3_unique")
                    new_quantity = st.number_input("Quantity", min_value=1, value=10, key="new_quantity_tab3_unique")
                    new_unit = st.selectbox("Unit", ["tons", "quintals", "kg", "dozen"], key="new_unit_tab3_unique")
                    new_price = st.number_input("Price per Unit (USD)", min_value=0.01, value=180.00, step=0.01, key="new_price_tab3_unique")
                    new_ready_by = st.date_input("Ready By", max(date.today() + timedelta(days=30), date(2025,6,1)), min_value=date.today(), key="new_ready_by_tab3_unique")
                    if st.form_submit_button("📜 List Produce"):
                        listing_id = f"ML{len(st.session_state.marketplace_listings) + 1:03d}"
                        st.session_state.marketplace_listings.append({
                            "id": listing_id, "produce": new_produce, "quantity": new_quantity, "unit": new_unit,
                            "price_per_unit": new_price, "ready_by": new_ready_by, "status": "Available"
                        })
                        add_to_ledger(f"New Listing: {new_produce} by {st.session_state.farmer_profile['farmverse_digital_id']}")
                        display_message(marketplace_msg_placeholder, "success", f"Listing for {new_produce} added!"); st.rerun()
    
    with col_smart_contracts: # Smart Contract Lifecycle
        with st.container(border=True):
            st.subheader("Smart Contract Lifecycle")
            sc = st.session_state.smart_contract
            st.metric("Current Contract Phase", sc["status"])
            st.progress(sc["progress"], text=f"Progress: {sc['progress']}%")
            if sc["current_contract_details"]:
                cd = sc["current_contract_details"]
                st.markdown(f"**Contract for:** {cd.get('produce')} ({cd.get('quantity')} {cd.get('unit')}) | **Value:** ${cd.get('quantity', 0) * cd.get('price_per_unit', 0):.2f}")
            else: st.info("No active contract.")
            contract_msg_placeholder = st.empty()

        st.subheader("Initiate Contract from Listings")
        for idx, listing in enumerate(st.session_state.marketplace_listings):
            if listing['status'] == 'Available':
                if st.button(f"🤝 Create Contract for {listing['produce']} (ID: {listing['id']})", key=f"create_contract_{listing['id']}_{idx}_tab3_unique"):
                    if sc["status"] == "Idle":
                        sc.update({
                            "status": "Agreed", "progress": 20, "current_contract_details": listing.copy(),
                            "active_contracts": sc["active_contracts"] + 1, "quality_inputs": {}, "current_quality_report": None
                        })
                        add_to_ledger(f"Smart Contract Agreed: {listing['id']}"); display_message(contract_msg_placeholder, "success", f"Contract for {listing['produce']} agreed!"); st.rerun()
                    else: display_message(contract_msg_placeholder, "warning", "Another contract active.")
        
        st.subheader("Contract Fulfillment Steps:")
        col_fund_tab3, col_delivery_tab3 = st.columns(2)
        with col_fund_tab3:
            if st.button("1️⃣ Fund Escrow", key="fund_escrow_tab3_unique", disabled=(sc["status"] != "Agreed")):
                if sc["current_contract_details"]:
                    amount = sc["current_contract_details"].get("quantity",0) * sc["current_contract_details"].get("price_per_unit",0)
                    if st.session_state.wallets["buyer_wallet_balance"] >= amount:
                        st.session_state.wallets["buyer_wallet_balance"] -= amount
                        st.session_state.wallets["escrow_wallet_balance"] += amount
                        sc.update({"status": "Escrowed", "progress": 40})
                        add_to_ledger(f"Escrow Funded: ${amount:.2f}"); display_message(contract_msg_placeholder, "success", "Escrow funded!"); st.rerun()
                    else: display_message(contract_msg_placeholder, "error", "Buyer balance low.")
        with col_delivery_tab3:
            if st.button("2️⃣ Confirm Delivery", key="confirm_delivery_tab3_unique", disabled=(sc["status"] != "Escrowed")):
                sc["show_logistics_form"] = True; display_message(contract_msg_placeholder, "info", "Fill logistics details."); st.rerun()

        if sc["show_logistics_form"]:
            with st.form("logistics_form_tab3_unique"):
                st.subheader("Logistics Details")
                sc_details = sc['current_contract_details']
                st.write(f"Delivery for: **{sc_details.get('produce')}**")
                log_from = st.text_input("Origin", st.session_state.farmer_profile['location'], key="log_from_tab3_unique")
                log_to = st.text_input("Destination", "Buyer Warehouse", key="log_to_tab3_unique")
                # Add other logistics inputs like quantity, date, notes if needed
                if st.form_submit_button("✅ Confirm Logistics"):
                    sc.update({"status": "Delivered", "progress": 60, "show_logistics_form": False})
                    # Store logistics_details if needed: sc["logistics_details"] = {...}
                    add_to_ledger(f"Delivery Confirmed: {sc_details['produce']}"); display_message(contract_msg_placeholder, "success", "Delivery confirmed!"); st.rerun()


        col_quality_val, col_release_payment = st.columns(2)
        with col_quality_val:
            if st.button("3️⃣ Validate Quality", key="validate_quality_btn_tab3_unique", disabled=(sc["status"] != "Delivered")):
                if sc["status"] == "Delivered":
                    sc["show_iot_quality_check"] = True
                    product_name = sc['current_contract_details'].get('produce')
                    if product_name and product_name not in sc.get("quality_inputs", {}):
                        sc.setdefault("quality_inputs", {})[product_name] = {}
                    display_message(contract_msg_placeholder, "info", "Initiating quality check."); st.rerun()

        if sc.get("show_iot_quality_check", False) and sc["current_contract_details"]: # Ensure contract details exist
            with st.container(border=True):
                st.subheader(f"🔍 Quality Validation: {sc['current_contract_details'].get('produce', 'N/A')}")
                st.info("Quality standards are based on verified data from agricultural research institutes like IARI.")
                
                uploaded_quality_image = st.file_uploader("Upload Quality Standard Chart / Produce Image (Optional)", type=["png", "jpg", "jpeg"], key=f"quality_image_upload_{sc['current_contract_details'].get('id','default')}")
                if uploaded_quality_image:
                    try: image = Image.open(uploaded_quality_image); st.image(image, caption="Uploaded Quality Reference", width=300)
                    except Exception as e: st.error(f"Error displaying image: {e}")
                
                product_name = sc['current_contract_details'].get('produce')
                specific_standards = QUALITY_STANDARDS.get(product_name, {})
                
                if not specific_standards: st.warning(f"No quality standards defined for '{product_name}'.")
                else:
                    quality_data_for_validation = {}
                    param_results = {} 
                    for param, details in specific_standards.items():
                        param_key = get_quality_input_key(product_name, param)
                        param_source = details.get("source")
                        ideal_value_str = ""
                        if "ideal_min" in details and "ideal_max" in details: ideal_value_str = f"Ideal: {details['ideal_min']} - {details['ideal_max']} {details.get('unit', '')}"
                        elif "min_ideal" in details: ideal_value_str = f"Ideal Min: {details['min_ideal']} {details.get('unit', '')}"
                        elif "max_allowed" in details: ideal_value_str = f"Max Allowed: {details['max_allowed']} {details.get('unit', '')}"
                        elif "max_allowed_area" in details: ideal_value_str = f"Max Allowed: {details['max_allowed_area']} {details.get('unit', '')}"
                        
                        current_value_input = None 
                        param_display_col, input_col = st.columns([2,3])
                        with param_display_col:
                            st.markdown(f"**{param}**")
                            if ideal_value_str: st.caption(ideal_value_str)
                            if details.get('note'): st.caption(f"Note: {details['note']}")
                        
                        with input_col:
                            if param_source == "iot":
                                iot_key = details.get("iot_key")
                                current_value_input = st.session_state.iot_data.get(iot_key)
                                st.write(f"*IoT Value ({iot_key}):* `{current_value_input}`")
                            elif param_source == "manual_input":
                                default_val = sc["quality_inputs"].get(product_name, {}).get(param, float(details.get("ideal_min", details.get("max_allowed",0.0)/2)))
                                current_value_input = st.number_input(f"Enter value ({details.get('unit','')})", value=float(default_val), key=param_key, step=0.01, label_visibility="collapsed")
                            elif param_source == "manual_input_select":
                                options = details.get("options", [0,1])
                                default_param_val = sc["quality_inputs"].get(product_name, {}).get(param, options[0])
                                default_idx_select = options.index(default_param_val) if default_param_val in options else 0
                                current_value_input = st.selectbox(f"Select for {param}", options, index=default_idx_select, key=param_key, label_visibility="collapsed")
                            elif param_source == "lab_report_placeholder":
                                passed_lab = st.checkbox(f"Assume passed (Lab Report)", value=sc["quality_inputs"].get(product_name, {}).get(param, True), key=param_key)
                                current_value_input = "Passed" if passed_lab else "Failed"
                            sc["quality_inputs"].setdefault(product_name, {})[param] = current_value_input
                        
                        quality_data_for_validation[param] = current_value_input
                        param_passed_validation = True; issue_for_param = ""
                        if current_value_input is None and param_source != "lab_report_placeholder": param_passed_validation = False; issue_for_param = "Missing value."
                        elif param_source == "lab_report_placeholder" and current_value_input == "Failed": param_passed_validation = False; issue_for_param = "Lab report: Failed."
                        elif param_source != "lab_report_placeholder":
                            try:
                                val_float = float(current_value_input)
                                if "ideal_min" in details and "ideal_max" in details and not (details["ideal_min"] <= val_float <= details["ideal_max"]): param_passed_validation = False; issue_for_param = f"Out of range."
                                elif "min_ideal" in details and val_float < details["min_ideal"]: param_passed_validation = False; issue_for_param = f"Below min."
                                elif "max_allowed" in details and val_float > details["max_allowed"]: param_passed_validation = False; issue_for_param = f"Exceeds max."
                            except: param_passed_validation = False; issue_for_param = "Invalid input."
                        param_results[param] = param_passed_validation
                        with param_display_col: 
                            st.markdown(f"{'✅' if param_passed_validation else '❌'}", unsafe_allow_html=True)
                        if not param_passed_validation: st.caption(f"Issue: {issue_for_param}")
                        st.markdown("---")
                    
                    passed_params_count = sum(1 for p_name, passed in param_results.items() if passed)
                    total_params_count = len(specific_standards)
                    
                    if total_params_count > 0: st.progress(passed_params_count / total_params_count, text=f"{passed_params_count}/{total_params_count} parameters passed")
                    
                    sc["current_quality_report"] = {"product": product_name, "data": quality_data_for_validation.copy(), "standards": specific_standards.copy()}
                    quality_met_overall = all(param_results.values()) if total_params_count > 0 else True 
                    issues = [f"{p_name} did not meet standards." for p_name, passed in param_results.items() if not passed]

                    radar_labels = []
                    radar_values_actual = []
                    radar_values_ideal = [] 

                    for param, details in specific_standards.items():
                        actual_val = quality_data_for_validation.get(param)
                        if details.get("source") not in ["iot", "manual_input", "visual_check", "manual_input_select"]: continue 
                        if actual_val is None or isinstance(actual_val, str) : continue 
                        try:
                            actual_val_float = float(actual_val)
                            radar_labels.append(param)
                            score = 0.5 
                            if "max_allowed" in details:
                                score = 1 - min(1, actual_val_float / details["max_allowed"]) if details["max_allowed"] > 0 else (1 if actual_val_float == 0 else 0)
                            elif "min_ideal" in details:
                                score = min(1, actual_val_float / details["min_ideal"]) if details["min_ideal"] > 0 else (1 if actual_val_float >=0 else 0)
                            elif "ideal_min" in details and "ideal_max" in details:
                                if details["ideal_min"] <= actual_val_float <= details["ideal_max"]: score = 1.0
                                else: 
                                    mid_point = (details["ideal_min"] + details["ideal_max"]) / 2
                                    range_width = details["ideal_max"] - details["ideal_min"]
                                    if range_width > 0: score = 1 - min(1, abs(actual_val_float - mid_point) / (range_width / 2))
                                    else: score = 1 if actual_val_float == mid_point else 0
                            radar_values_actual.append(max(0,min(1,score)) * 100) 
                            radar_values_ideal.append(100) 
                        except (ValueError, TypeError): continue 
                    
                    if radar_labels and len(radar_labels) >=3 : 
                        fig_radar = go.Figure()
                        fig_radar.add_trace(go.Scatterpolar(r=radar_values_actual + [radar_values_actual[0]], theta=radar_labels + [radar_labels[0]], fill='toself', name='Actual Quality Score (%)'))
                        fig_radar.add_trace(go.Scatterpolar(r=radar_values_ideal + [radar_values_ideal[0]], theta=radar_labels + [radar_labels[0]], mode='lines', name='Ideal (100%)', line=dict(dash='dot', color='grey')))
                        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), title="Quality Metrics Radar Chart", height=400)
                        st.plotly_chart(fig_radar, use_container_width=True)
                    elif radar_labels: st.info("Not enough numerical parameters (need at least 3) to display radar chart.")

                    if quality_met_overall:
                        st.success("✅ All specified quality parameters met!")
                        if st.button("Confirm Quality Validation", key="confirm_quality_final_tab3_unique"):
                            sc.update({"status": "Quality Validated", "progress": 80, "show_iot_quality_check": False})
                            st.session_state.farmer_profile['successful_validations_count'] += 1
                            award_badge_if_eligible("Quality King", st.session_state.farmer_profile['successful_validations_count'] >= BADGE_DEFINITIONS["Quality King"]["threshold_validations"], st.session_state.farmer_profile)
                            st.session_state.wallets['fvt_balance'] += FVT_REWARD_QUALITY_VALIDATION
                            st.toast(f"💰 +{FVT_REWARD_QUALITY_VALIDATION} FVT!", icon="🪙")
                            add_to_ledger(f"Quality Validated: {product_name}. FVT awarded."); display_message(contract_msg_placeholder, "success", "Quality validated!"); st.rerun()
                    else:
                        st.error("⚠️ Some quality parameters not met:"); 
                        for issue in issues: st.write(f"- {issue}")
                        st.warning("You can choose to override or raise a dispute.")
                        col_ovr, col_dsp = st.columns(2)
                        with col_ovr: 
                            if st.button("Override & Validate", key="override_quality_tab3_unique"):
                                sc.update({"status": "Quality Validated", "progress": 80, "show_iot_quality_check": False})
                                add_to_ledger(f"Quality Overridden: {product_name}"); display_message(contract_msg_placeholder,"warning", "Quality overridden!"); st.rerun()
                        with col_dsp:
                             if st.button("🚨 Raise Dispute", key="raise_dispute_quality_tab3_unique"):
                                sc.update({"status": "In Dispute", "in_dispute": sc["in_dispute"]+1, "active_contracts": max(0, sc["active_contracts"]-1), "show_iot_quality_check": False, "current_contract_details":{}, "progress":0, "current_quality_report":None})
                                add_to_ledger(f"Contract Disputed: {product_name}"); display_message(contract_msg_placeholder, "error", "Contract disputed!"); st.rerun()

        with col_release_payment:
            if st.button("4️⃣ Release Payment", key="release_payment_tab3_unique", disabled=(sc["status"] != "Quality Validated")):
                if sc["status"] == "Quality Validated":
                    amount = st.session_state.wallets["escrow_wallet_balance"]
                    if amount > 0:
                        st.session_state.wallets["farmer_wallet_balance"] += amount
                        st.session_state.wallets["escrow_wallet_balance"] = 0.00
                        sc.update({"status": "Completed", "progress": 100, "last_payment_status": "Successful", 
                                   "completed_successful": sc["completed_successful"] + 1, 
                                   "active_contracts": max(0, sc["active_contracts"] -1)})
                        st.session_state.farmer_profile['flawless_contracts_count'] +=1
                        award_badge_if_eligible("Contract Master", st.session_state.farmer_profile['flawless_contracts_count'] >= BADGE_DEFINITIONS["Contract Master"]["threshold_contracts"], st.session_state.farmer_profile)
                        
                        st.session_state.marketplace_metrics["sales_volume"] += amount
                        st.session_state.marketplace_metrics["transactions_count"] += 1
                        if sc["current_contract_details"] and "id" in sc["current_contract_details"]:
                            for i, lst in enumerate(st.session_state.marketplace_listings):
                                if lst["id"] == sc["current_contract_details"]["id"]: st.session_state.marketplace_listings[i]["status"] = "Sold"; break
                        add_to_ledger(f"Payment Released: ${amount:.2f}"); display_message(contract_msg_placeholder, "success", "Payment released! 🎉"); 
                        sc.update({"current_contract_details": {}, "status": "Idle", "progress": 0, "quality_inputs": {}, "current_quality_report": None})
                        calculate_credit_score(); st.rerun()
                    else: display_message(contract_msg_placeholder, "error", "Escrow empty.")
        
        st.markdown("---")
        with st.container(border=True):
            st.subheader("Wallet Balances")
            col_fw_tab3, col_fvt_tab3, col_ew_tab3, col_bw_tab3 = st.columns(4)
            with col_fw_tab3: st.metric("👨‍🌾 Farmer (USD)", f"${st.session_state.wallets['farmer_wallet_balance']:.2f}")
            with col_fvt_tab3: st.metric("🪙 Farmer (FVT)", f"{st.session_state.wallets.get('fvt_balance',0.0):.2f} FVT")
            with col_ew_tab3: st.metric("🔒 Escrow (USD)", f"${st.session_state.wallets['escrow_wallet_balance']:.2f}")
            with col_bw_tab3: st.metric("🛍️ Buyer (USD)", f"${st.session_state.wallets['buyer_wallet_balance']:.2f}")
        
        st.markdown("---")
        with st.container(border=True):
            st.subheader("Recent Blockchain Ledger Activity")
            if st.session_state.blockchain_ledger:
                ledger_df = pd.DataFrame(st.session_state.blockchain_ledger).sort_values(by="block_id", ascending=False)
                ledger_df['timestamp'] = pd.to_datetime(ledger_df['timestamp']).dt.strftime("%Y-%m-%d %H:%M:%S")
                ledger_df_display = ledger_df[['block_id', 'timestamp', 'prev_hash', 'current_hash', 'transaction']].copy()
                ledger_df_display['prev_hash'] = ledger_df_display['prev_hash'].apply(lambda x: x[:8] + "..." + x[-8:] if len(x) > 16 else x)
                ledger_df_display['current_hash'] = ledger_df_display['current_hash'].apply(lambda x: x[:8] + "..." + x[-8:] if len(x) > 16 else x)
                st.dataframe(ledger_df_display, height=300, hide_index=True, use_container_width=True)
            else: st.info("No blockchain activity.")
            st.caption(f"Next Block ID: {st.session_state.next_block_id}")

#--- Tab 4: Carbon Credits ---
with tab4:
    st.header("🌿 Carbon Credits: Earn for Sustainability")
    carbon_credit_msg_placeholder = st.empty()
    col_cc_overview, col_cc_verification = st.columns(2)
    with col_cc_overview:
        with st.container(border=True):
            st.subheader("Your Carbon Credit Overview")
            cc = st.session_state.carbon_credits
            col_e, col_s, col_p, col_m = st.columns(4)
            col_e.metric("Total Earned", f"{float(cc['earned_total']):.2f} CC")
            col_s.metric("Total Sold", f"{float(cc['sold_total']):.2f} CC")
            col_p.metric("Pending Mint", f"{float(cc['pending_mint']):.2f} CC")
            col_m.metric("Market Price", f"${float(cc['market_price_per_credit']):.2f}/CC")
            if st.button("♻️ Recalculate Carbon Credits", key="recalc_cc_btn_tab4_unique"):
                calculate_carbon_credits()
                cc.update({"verification_status": "Pending Verification", "verified_credits": 0.0})
                display_message(carbon_credit_msg_placeholder, "info", f"Credits recalculated. {float(cc['pending_mint']):.2f} pending verification.")
                st.rerun()
        with st.container(border=True):
            st.subheader("Carbon Credit Calculation Impact")
            if not st.session_state.carbon_credit_feature_impact.empty:
                fig_impact_cc = px.bar(st.session_state.carbon_credit_feature_impact.head(5), x='Importance', y='Feature', orientation='h', title="Top Parameters for Carbon Credits", color='Importance', color_continuous_scale=px.colors.sequential.Greens)
                fig_impact_cc.update_layout(yaxis={'categoryorder':'total ascending'}); st.plotly_chart(fig_impact_cc, use_container_width=True)
            else: st.info("Recalculate credits to see impact.")

    with col_cc_verification:
        with st.container(border=True):
            st.subheader("Carbon Credits Distribution")
            cc_data_list = [{"Category": k.replace("_total","").replace("_"," ").title(), "Credits": float(v)} for k,v in cc.items() if k in ["earned_total", "sold_total", "pending_mint", "verified_credits"]]
            cc_df = pd.DataFrame([item for item in cc_data_list if item["Credits"] > 0 or item["Category"] in ["Pending Mint", "Verified Credits"]])
            if not cc_df.empty:
                fig_cc_dist = px.bar(cc_df, x="Category", y="Credits", title="Carbon Credits Status", color="Category", template="plotly_white", color_discrete_map={"Earned Total": "#28a745", "Sold Total": "#007bff", "Pending Mint": "#ffc107", "Verified Credits": "#17a2b8"})
                st.plotly_chart(fig_cc_dist, use_container_width=True)
            else: st.info("No carbon credit data to display yet.")

        with st.container(border=True):
            st.subheader("Carbon Credit Verification Process")
            st.metric("Verification Status", cc['verification_status'])
            st.metric("Verified Credits Available", f"{float(cc['verified_credits']):.2f} CC")
            st.info(f"Verification by: **{cc['verification_org']}**")

            if cc['verification_status'] == "Pending Verification":
                if st.button("🔬 Simulate Verification by Verra", key="simulate_verra_btn_tab4_unique", disabled=(float(cc['pending_mint']) <=0)):
                    if random.random() > 0.1: 
                        cc.update({"verification_status": "Verified", "verified_credits": float(cc['pending_mint']), "pending_mint": 0.0})
                        st.session_state.wallets['fvt_balance'] += FVT_REWARD_CARBON_VERIFICATION
                        st.toast(f"💰 +{FVT_REWARD_CARBON_VERIFICATION} FVT for carbon credit verification!", icon="🪙")
                        add_to_ledger(f"Carbon Credits Verified: {float(cc['verified_credits']):.2f}. FVT awarded.")
                        display_message(carbon_credit_msg_placeholder, "success", "Credits verified! FVT awarded.")
                    else:
                        cc.update({"verification_status": "Rejected", "verified_credits": 0.0})
                        add_to_ledger("Carbon Credits Verification Rejected."); display_message(carbon_credit_msg_placeholder, "error", "Verification rejected.")
                    st.rerun()
            elif cc['verification_status'] == "Verified":
                if st.button("⛏️ Mint All Verified Credits", key="mint_verified_cc_btn_tab4_unique", disabled=(float(cc['verified_credits']) <=0)):
                    credits_to_mint = float(cc['verified_credits'])
                    cc.update({"earned_total": float(cc['earned_total']) + credits_to_mint, "verified_credits": 0.0, "verification_status": "Minted"})
                    st.session_state.farmer_profile['co2_offset_total'] += credits_to_mint 
                    award_badge_if_eligible("Eco Warrior", st.session_state.farmer_profile['co2_offset_total'] >= BADGE_DEFINITIONS["Eco Warrior"]["threshold_cc_earned"], st.session_state.farmer_profile)
                    add_to_ledger(f"Carbon Credits Minted: {credits_to_mint:.2f}")
                    display_message(carbon_credit_msg_placeholder, "success", f"{credits_to_mint:.2f} CC Minted!"); calculate_credit_score(); st.rerun()
            elif cc['verification_status'] == "Rejected":
                if st.button("↩️ Reset Verification", key="reset_ver_cc_btn_tab4_unique"):
                    cc.update({"verification_status":"Pending Calculation", "verified_credits":0.0}); st.rerun()
            
            available_to_sell_cc = float(cc['earned_total']) - float(cc['sold_total'])
            if available_to_sell_cc > 0 and cc['verification_status'] == "Minted":
                st.subheader("Sell Your Carbon Credits")
                sell_qty_cc = st.number_input("Quantity to Sell (CC)", 0.01, float(max(0.01, available_to_sell_cc)), float(min(1.0, max(0.01,available_to_sell_cc))), 0.01, key="sell_cc_qty_tab4_unique", disabled=(available_to_sell_cc<=0))
                if st.button(f"💰 Sell {float(sell_qty_cc):.2f} CC", key="sell_cc_btn_tab4_unique", disabled=(available_to_sell_cc<=0 or float(sell_qty_cc)<=0)):
                    cc['sold_total'] = float(cc['sold_total']) + float(sell_qty_cc)
                    earnings_cc = float(sell_qty_cc) * float(cc['market_price_per_credit'])
                    st.session_state.wallets['farmer_wallet_balance'] += earnings_cc
                    add_to_ledger(f"Carbon Credits Sold: {float(sell_qty_cc):.2f} for ${earnings_cc:.2f}")
                    display_message(carbon_credit_msg_placeholder, "success", f"{float(sell_qty_cc):.2f} CC Sold for ${earnings_cc:.2f}!"); calculate_credit_score(); st.rerun()
            elif cc['verification_status'] != "Minted" and available_to_sell_cc > 0 : st.info("Mint your verified credits to make them available for selling.")
            elif available_to_sell_cc <= 0 : st.info("No earned carbon credits available to sell.")

    st.markdown("---")
    with st.container(border=True):
        st.subheader("🏆 Community Carbon Champions (Leaderboard)")
        st.info("A leaderboard showcasing top carbon contributors will be featured here in future updates.")
        st.write(f"Your Total CO₂ Offset (from earned credits): **{st.session_state.farmer_profile.get('co2_offset_total', 0.0):.2f} CC** (approx. {st.session_state.farmer_profile.get('co2_offset_total', 0.0) * 25:.0f} kg CO₂e)")


#--- Tab 5: Al Credit Score & Financial Linkages ---
with tab5:
    st.header("💰 AI-Driven Credit Scoring & Financial Linkages")
    st.markdown("### Access Fairer Credit Based on Your Real-time Farm Performance and Reliability")
    credit_score_msg_placeholder = st.empty()
    col_score_display, col_financial_linkages = st.columns(2)
    with col_score_display:
        with st.container(border=True):
            st.subheader("Overall FarmVerse Credit Score")
            display_score = max(0, min(800, int(st.session_state.overall_credit_score)))
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = display_score, domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Farm Credit Score (0-800)"},
                gauge = {'axis': {'range': [None, 800], 'tickwidth': 1, 'tickcolor': "darkblue"}, 
                         'bar': {'color': "#0e8c4f"}, 
                         'steps': [{'range': [0, 400], 'color': "#dc3545"}, {'range': [400, 600], 'color': "#ffc107"}, {'range': [600, 800], 'color': "#28a745"}],
                         'threshold': {'line': {'color': "darkorange", 'width': 4}, 'thickness': 0.75, 'value': 650}}))
            fig_gauge.update_layout(height=250, margin=dict(l=10,r=10,t=50,b=10)); st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown("##### Factors Influencing Your Score (0-100 Scale):")
            factors_df_data = [{'Factor': k.replace('_',' ').title(), 'Impact Score': f"{float(v):.1f}"} for k,v in st.session_state.credit_score_factors.items()]
            st.dataframe(pd.DataFrame(factors_df_data), use_container_width=True, hide_index=True)
        with st.container(border=True):
            st.subheader("Simulate Resource Sharing Activity")
            new_res_score = st.slider("Resource Sharing Engagement (0-100)", 0, 100, int(st.session_state.resource_sharing_activity_score), 1, key="res_share_slider_tab5_unique")
            if new_res_score != st.session_state.resource_sharing_activity_score:
                st.session_state.resource_sharing_activity_score = float(new_res_score); calculate_credit_score()
                add_to_ledger("Credit Score: Resource Sharing Update"); display_message(credit_score_msg_placeholder, "info", "Credit score updated."); st.rerun()
        with st.container(border=True):
            st.subheader("Areas for Credit Score Improvement")
            if st.session_state.credit_improvement_suggestions:
                for sug in st.session_state.credit_improvement_suggestions: st.info(f"💡 {sug}")
            else: st.success("Credit profile excellent!")
    with col_financial_linkages:
        with st.container(border=True):
            st.subheader("Secure Profile Sharing with Lenders")
            st.markdown("With explicit consent, share your verified FarmVerse credit profile to unlock financial products.")
            st.write(f"**Partners:** {', '.join(st.session_state.financial_linkage_partners)}")
            if st.session_state.credit_profile_shared_status:
                st.success("✅ Profile SHARED with lenders.")
                if st.button("🛑 Stop Sharing", key="stop_share_tab5_unique"): 
                    st.session_state.credit_profile_shared_status=False; add_to_ledger("Profile Sharing Stopped"); st.rerun()
            else:
                st.warning("⚠️ Profile NOT SHARED.")
                if st.button("🤝 Share Profile", key="share_profile_tab5_unique"):
                    st.session_state.credit_profile_shared_status=True; add_to_ledger("Profile Shared"); st.rerun()
        with st.container(border=True):
            st.subheader("Contract Performance Metrics")
            sc_m = st.session_state.smart_contract
            c1,c2,c3 = st.columns(3)
            c1.metric("Active", sc_m['active_contracts'])
            c2.metric("Completed", sc_m['completed_successful'])
            c3.metric("Disputed", sc_m['in_dispute'])
            st.metric("Last Payment", sc_m['last_payment_status'])
        with st.container(border=True):
            st.subheader("Available Loans & Subsidies")
            loan_msg_ph_tab5 = st.empty() # Unique placeholder
            for idx, loan in enumerate(st.session_state.available_loans):
                st.markdown(f"**{loan['name']}** (ID: {loan['id']})")
                st.write(f"- Rate: {loan['interest_rate']}, Max: {loan['max_amount']}, Min Score: {loan['eligibility_score']}, App Status: {loan['application_status']}")
                lc1, lc2 = st.columns(2)
                with lc1: 
                    if st.button("🔍 Verify Eligibility", key=f"verify_loan_{loan['id']}_{idx}_tab5_unique"):
                        if float(st.session_state.overall_credit_score) >= loan['eligibility_score']: display_message(loan_msg_ph_tab5, "success", f"Eligible for {loan['name']}!")
                        else: display_message(loan_msg_ph_tab5, "warning", f"Not eligible. Need score {loan['eligibility_score']}.")
                with lc2:
                    apply_disabled = not (float(st.session_state.overall_credit_score) >= loan['eligibility_score'] and st.session_state.credit_profile_shared_status and loan['application_status']=="N/A")
                    if st.button("📝 Apply Now", key=f"apply_loan_{loan['id']}_{idx}_tab5_unique", disabled=apply_disabled):
                        st.session_state.available_loans[idx].update({"status":"Applied", "application_status":"Under Review"})
                        add_to_ledger(f"Applied for {loan['name']}"); display_message(loan_msg_ph_tab5, "success", f"Applied for {loan['name']}."); st.rerun()
                if loan['application_status'] == "Under Review":
                    if st.button(f"🚀 Simulate {loan['name']} Approval", key=f"sim_approve_loan_{loan['id']}_{idx}_tab5_unique"):
                        st.session_state.available_loans[idx]['application_status'] = "Approved & Disbursed"
                        try:
                            amt_str = loan['max_amount'].replace('$','').replace(',',''); amt_val = float(amt_str)
                            st.session_state.wallets['farmer_wallet_balance'] += amt_val
                            add_to_ledger(f"Loan Approved: {loan['name']} - ${amt_val:.2f}"); display_message(loan_msg_ph_tab5, "success", f"{loan['name']} approved for ${amt_val:.2f}!"); st.rerun()
                        except ValueError: display_message(loan_msg_ph_tab5, "error", "Error processing loan amount.")
                st.markdown("---")

#--- Tab 6: Analytics & Reports ---
with tab6:
    st.header("📈 Comprehensive Analytics & Reports")
    st.markdown("### A Single-Shot Overview of Your FarmVerse Performance and Impact")
    st.subheader("Overall FarmVerse Performance Summary")
    col_s1, col_s2, col_s3 = st.columns(3)
    fp = st.session_state.farmer_profile; cc = st.session_state.carbon_credits; wallets = st.session_state.wallets; sc_m = st.session_state.smart_contract; dao = st.session_state.dao
    with col_s1:
        with st.container(border=True, height=280):
            st.markdown(f"<p class='metric-label'>Farmer ID</p><p class='metric-value'>{fp['farmverse_digital_id']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-label'>Land Size</p><p class='metric-value'>{fp['land_holding_size']:.1f} Acres</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-label'>Badges</p><p class='metric-value'>{len(fp.get('badges',[]))}</p>", unsafe_allow_html=True)
    with col_s2:
        with st.container(border=True, height=280):
            st.markdown(f"<p class='metric-label'>Credit Score</p><p class='metric-value'>{float(st.session_state.overall_credit_score):.0f}/800</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-label'>CO₂ Offset (Earned CC)</p><p class='metric-value'>{float(fp.get('co2_offset_total',0.0)):.2f} CC</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-label'>Loyalty Points</p><p class='metric-value'>{fp.get('loyalty_points',0)}</p>", unsafe_allow_html=True)
    with col_s3:
        with st.container(border=True, height=280):
            st.markdown(f"<p class='metric-label'>Wallet (USD)</p><p class='metric-value'>${wallets['farmer_wallet_balance']:.2f}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-label'>Wallet (FVT)</p><p class='metric-value'>{wallets.get('fvt_balance',0.0):.2f} FVT</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-label'>Completed Contracts</p><p class='metric-value'>{sc_m['completed_successful']}</p>", unsafe_allow_html=True)

    st.markdown("---")
    col_iot_sum_tab6, col_adv_sum_tab6 = st.columns(2)
    with col_iot_sum_tab6:
        with st.container(border=True):
            st.subheader("Current IoT Sensor Readings")
            iot_d = st.session_state.iot_data
            st.write(f"**Soil Moisture:** {iot_d['soil_moisture']:.1f}%"); st.write(f"**Grain Moisture:** {iot_d.get('grain_moisture_content', 'N/A'):.1f}%") 
            st.write(f"**Avg Temp:** {iot_d['avg_temp']:.1f}°C"); st.write(f"**Humidity:** {iot_d['humidity']:.1f}%"); st.write(f"**Soil pH:** {iot_d['soil_pH']:.1f}")
    with col_adv_sum_tab6:
        with st.container(border=True):
            st.subheader("Key Advisory Insights")
            adv_o = st.session_state.advisory_output
            st.write(f"**Irrigation:** {adv_o.get('irrigation_needed', 'N/A')}"); st.write(f"**Pest Risk:** {adv_o.get('pest_risk', 'N/A')}"); st.write(f"**Yield Est.:** {adv_o.get('yield_prediction', 'N/A')}")
    
    st.markdown("---")
    with st.container(border=True):
        st.subheader("Historical AI Advisory Trends")
        if st.session_state.advisory_history:
            advisory_plotting_data = []
            for entry_date_hist, advisory_dict_hist in st.session_state.advisory_history:
                try:
                    irrigation_val = 0
                    if "High irrigation needed." in advisory_dict_hist.get('irrigation_needed',''): irrigation_val = 2
                    elif "Moderate irrigation needed." in advisory_dict_hist.get('irrigation_needed',''): irrigation_val = 1
                    pest_risk_val = 0
                    if "High pest risk!" in advisory_dict_hist.get('pest_risk',''): pest_risk_val = 2
                    elif "Moderate pest risk." in advisory_dict_hist.get('pest_risk',''): pest_risk_val = 1
                    yield_pred_str = advisory_dict_hist.get('yield_prediction', '0').replace('Estimated yield: ', '').replace(' T/Acre.', '')
                    yield_val = float(yield_pred_str) if yield_pred_str and yield_pred_str != "N/A" else 0.0
                    advisory_plotting_data.append({
                        "Date": entry_date_hist, "Irrigation Need (0=No, 1=Mod, 2=High)": irrigation_val,
                        "Pest Risk (0=Low, 1=Mod, 2=High)": pest_risk_val, "Yield Prediction (T/Acre)": yield_val
                    })
                except ValueError as e: st.warning(f"Parse error for yield on {entry_date_hist}: {e}"); continue 
            if advisory_plotting_data:
                advisory_df = pd.DataFrame(advisory_plotting_data).sort_values(by='Date')
                for col_name, title_suffix in [("Irrigation Need (0=No, 1=Mod, 2=High)","Irrigation Needs"), ("Pest Risk (0=Low, 1=Mod, 2=High)","Pest Risk"), ("Yield Prediction (T/Acre)","Yield Prediction")]:
                    fig = px.line(advisory_df, x="Date", y=col_name, title=f"Historical {title_suffix}", template="plotly_white")
                    if "Need" in col_name or "Risk" in col_name : fig.update_yaxes(tickvals=[0,1,2], ticktext=["Low/No","Moderate","High"])
                    st.plotly_chart(fig, use_container_width=True)
            else: st.info("No valid historical advisory data to plot.")
        else: st.info("No historical advisory data yet. Update IoT data to generate advisories.")

    st.markdown("---")
    with st.container(border=True):
        st.subheader("Recent Blockchain Activity (Last 5)")
        if st.session_state.blockchain_ledger:
            ledger_df_summary = pd.DataFrame(st.session_state.blockchain_ledger)
            ledger_df_summary['timestamp'] = \
                pd.to_datetime(ledger_df_summary['timestamp']).dt.strftime("%Y-%m-%d %H:%M:%S")
            ledger_df_summary_display = ledger_df_summary[['block_id', 'timestamp', 'transaction']].copy()
            st.dataframe(ledger_df_summary_display.sort_values(by="block_id", ascending=False).head(5), use_container_width=True, hide_index=True)
        else: st.info("No blockchain activity recorded yet.")

    st.markdown("---")
    col_compliance, col_credit_focus_report = st.columns(2) 
    with col_compliance:
        with st.container(border=True):
            st.subheader("Sustainable Practices Compliance")
            total_sustainable_tasks_report = sum(1 for t in st.session_state.todays_tasks if t.get('sustainable_practice'))
            completed_sustainable_tasks_report = sum(1 for t in st.session_state.todays_tasks if t['done'] and t.get('sustainable_practice'))
            if total_sustainable_tasks_report > 0:
                compliance_percentage = (completed_sustainable_tasks_report / total_sustainable_tasks_report) * 100
                st.progress(compliance_percentage / 100, text=f"Sustainable Task Completion: **{compliance_percentage:.1f}%** ({completed_sustainable_tasks_report}/{total_sustainable_tasks_report} completed)")
            else: st.info("No sustainable tasks defined yet. Add some in the 'IoT Data & Al Advisory' tab!")
    with col_credit_focus_report:
        with st.container(border=True):
            st.subheader("Credit Score Improvement Focus Areas")
            if st.session_state.credit_improvement_suggestions:
                for suggestion in st.session_state.credit_improvement_suggestions: st.write(f"💡 {suggestion}")
            else: st.success("No specific improvement areas identified. Your credit profile is excellent! ")
