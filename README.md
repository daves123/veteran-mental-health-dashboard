# Female Veterans Mental Health Analysis - Streamlit Dashboard

## Overview
This interactive dashboard presents findings from the analysis of mental health disparities among female veterans using BRFSS 2024 data.

**Author:** Dave Singh  
**Course:** CS 7510 - Communication and Presentation for Data Science  
**Semester:** Fall 2025  
**Dataset:** BRFSS 2024 (~6,000 female veterans)

## Features

The dashboard includes five main sections:

1. **Overview** - Executive summary with key metrics and distributions
2. **Geographic Analysis** - State-level mental health burden analysis
3. **Key Insights** - Risk factors and protective factors analysis
4. **Predictive Modeling** - Machine learning model performance and feature importance
5. **Recommendations** - Evidence-based policy recommendations

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install streamlit==1.28.0 pandas==2.1.0 numpy==1.25.0 plotly==5.17.0 scikit-learn==1.3.0
```

### Step 2: Run the Application

```bash
streamlit run streamlit_app.py
```

The dashboard will automatically open in your default web browser at `http://localhost:8501`

## Using the Dashboard

### Navigation
- Use the sidebar radio buttons to navigate between different analysis sections
- Each section provides interactive visualizations and key insights
- Hover over charts for detailed information
- Click legend items to show/hide data series

### Key Metrics
The Overview page displays four critical metrics:
- Total female veteran sample size
- Depression diagnosis rate with comparison to male veterans
- Average poor mental health days per month
- Percentage experiencing frequent mental distress

### Interactive Visualizations
All charts are interactive:
- **Zoom**: Click and drag to zoom into specific areas
- **Pan**: Shift + click and drag to pan across the chart
- **Reset**: Double-click to reset zoom
- **Download**: Use the camera icon to download charts as PNG

## Project Structure

```
.
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── (Additional files from Phase 2 analysis)
```

## Data Notes

**Important:** This dashboard uses synthetic demonstration data that mimics the statistical patterns observed in the actual BRFSS 2024 dataset. For the complete analysis with real data, please refer to:
- `Phase2 Intermediate Report.docx` - Detailed written report

## Technical Details

### Data Generation
The app generates synthetic data with realistic distributions:
- Right-skewed mental health days distribution
- Depression prevalence of 46% (matching real data)
- Income and employment distributions reflecting veteran populations
- Geographic distribution across top veteran-population states

### Visualization Library
All charts use Plotly for interactive visualizations, providing:
- Responsive design
- Professional styling
- Export capabilities
- Mobile-friendly interface

## Deployment

### Run Locally
```bash
streamlit run streamlit_app.py
```

### Cloud Deployment Options

#### Streamlit Cloud
1. Push your code to GitHub
2. Go to https://share.streamlit.io/
3. Connect your GitHub repository
4. Deploy!

## Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError: No module named 'streamlit'`  
**Solution:** Ensure you've installed all dependencies: `pip install -r requirements.txt`

**Issue:** Port 8501 already in use  
**Solution:** Stop other Streamlit instances or specify a different port:  
```bash
streamlit run streamlit_app.py --server.port 8502
```

**Issue:** Charts not displaying  
**Solution:** Ensure you have a modern browser (Chrome, Firefox, Safari, or Edge)

## Acknowledgments

- **Data Source:** Centers for Disease Control and Prevention (CDC), Behavioral Risk Factor Surveillance System (BRFSS) 2024
- **Libraries:** Streamlit, Plotly, Pandas, NumPy, Scikit-learn
- **Inspiration:** The 2+ million female veterans who have served our nation

## License

This project is created for educational purposes as part of CS 7510 coursework.

---
