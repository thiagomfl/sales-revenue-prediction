"""
Streamlit frontend for Sales Revenue Prediction.

This module provides an interactive web interface for making
revenue predictions using the trained ML model.
"""

import streamlit as st
import requests

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

API_URL = "http://localhost:8000"

st.set_page_config(
  page_title='Sales Revenue Prediction', page_icon='📈',
  layout='centered', initial_sidebar_state='collapsed')

# -----------------------------------------------------------------------------
# Custom CSS
# -----------------------------------------------------------------------------

st.markdown(
  """
    <style>
      .main-header {
        color: #1E88E5;
        font-weight: 700;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
      }
      .sub-header {
        color: #666666;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
      }
      .prediction-box {
        margin: 2rem 0;
        padding: 2.5rem;
        text-align: center;
        border-radius: 1.25rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 40px rgba(15, 76, 92, 0.3);
        background: linear-gradient(135deg, #0F4C5C 0%, #1A936F 50%, #88D498 100%);
      }
      .prediction-label {
        color: #C6F7E2;
        font-weight: 600;
        font-size: 1.25rem;
        margin-bottom: 1rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
      }
      .prediction-value {
        color: #FFFFFF;
        font-weight: 800;
        line-height: 1.2;
        font-size: 3.5rem;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
      }
      .prediction-currency {
        opacity: 0.9;
        font-size: 2rem;
        font-weight: 600;
      }
      .info-card {
        margin: 1rem 0;
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #F8F9FA;
        border-left: 4px solid #1A936F;
      }
      .stButton > button {
        width: 100%;
        color: white;
        border: none;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
        background: linear-gradient(135deg, #0F4C5C 0%, #1A936F 100%);
      }
      .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(26, 147, 111, 0.4);
        background: linear-gradient(135deg, #1A936F 0%, #0F4C5C 100%);
      }
    </style>
  """,
  unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------

st.markdown('<p class="main-header">📈 Sales Revenue Prediction</p>', unsafe_allow_html=True)
st.markdown(
  '<p class="sub-header">Predict revenue based on seller experience, sales count, and seasonal factors</p>',
  unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar - Model Info
# -----------------------------------------------------------------------------

with st.sidebar:
  st.header('ℹ️ Model Information')
  
  try:
    response = requests.get(f'{API_URL}/api/v1/model/info', timeout=5)
    if response.status_code == 200:
      model_info = response.json()
      st.success('Model loaded successfully!')
      st.json(model_info)
    else:
      st.error('Failed to load model info')
  except requests.exceptions.ConnectionError:
    st.warning('API not available. Make sure the API is running.')
  except Exception as e:
    st.error(f'Error: {str(e)}')

# -----------------------------------------------------------------------------
# Input Form
# -----------------------------------------------------------------------------

st.markdown('### 📝 Enter Seller Information')

col1, col2 = st.columns(2)

with col1:
  experience_months = st.number_input(
    'Experience (months)', min_value=0, max_value=600, value=36,
    step=1, help='How many months has the seller been working?')

  seasonal_factor = st.slider(
    'Seasonal Factor', min_value=1, max_value=10, value=5,
    step=1, help='1 = Low season, 10 = Peak season')

with col2:
  number_of_sales = st.number_input(
    'Number of Sales', min_value=0, max_value=1000, value=50,
    step=1, help='How many sales has the seller made?')

  st.markdown('<br>', unsafe_allow_html=True)
  
  # Visual indicator for seasonal factor
  seasonal_labels = {
    1: '🥶 Very Low',
    2: '❄️ Low',
    3: '🌧️ Below Average',
    4: '🌥️ Slightly Low',
    5: '⛅ Average',
    6: '🌤️ Slightly High',
    7: '☀️ Above Average',
    8: '🔥 High',
    9: '🔥🔥 Very High',
    10: '🚀 Peak Season',
  }
  st.info(f'Season: {seasonal_labels[seasonal_factor]}')

# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------

st.markdown('---')

if st.button('🔮 Predict Revenue', use_container_width=True):
  with st.spinner('Calculating prediction...'):
    try:
      payload = {
        'experience_months': experience_months,
        'number_of_sales': number_of_sales,
        'seasonal_factor': seasonal_factor,
      }

      response = requests.post(f'{API_URL}/api/v1/predict', json=payload, timeout=10)

      if response.status_code == 200:
        result = response.json()
        predicted_revenue = result['predicted_revenue']

        # Format the value with thousands separator
        formatted_value = f'{predicted_revenue:,.2f}'

        # Display prediction
        st.markdown(
          f"""
            <div class="prediction-box">
              <p class="prediction-label">💰 Predicted Revenue</p>
              <p class="prediction-value">R$ {predicted_revenue:,.2f}</p>
            </div>
          """,
          unsafe_allow_html=True,
        )

        # Display input summary
        st.markdown('#### 📊 Input Summary')
        
        col1, col2, col3 = st.columns(3)
        col1.metric('Experience', f'{experience_months} months')
        col2.metric('Sales', f'{number_of_sales}')
        col3.metric('Season Factor', f'{seasonal_factor}/10')

      else:
        st.error(f'Prediction failed: {response.text}')

    except requests.exceptions.ConnectionError:
      st.error(
        '❌ Could not connect to the API. Make sure the API is running with `poetry run poe api`')
    except Exception as e:
      st.error(f'❌ An error occurred: {str(e)}')


def main() -> None:
  """
  Entry point for the Streamlit app.
  """
  pass  # Streamlit runs the script directly


if __name__ == '__main__':
  main()
