# **TempestAI: Multi-Model Weather Forecasting with Deep Learning**  
*Predicting Tomorrow’s Weather Today with Neural Networks*

TempestAI is a dynamic deep learning system designed for hourly temperature prediction using various neural network architectures. Compare the performance of GRU with attention, LSTM Encoder-Decoder, CNN-GRU hybrids, and dense models while leveraging cyclical feature engineering and automated weather data pipelines.

---

## **Features**

- **Multi-Model Comparison:** Evaluate four distinct architectures:
  - GRU with Attention
  - LSTM Encoder-Decoder
  - CNN-GRU Hybrid
  - Dense Model
- **Cyclical Feature Engineering:** Apply sine and cosine transformations for capturing hour, day, and weekday seasonalities.
- **API Integration:** Seamless data retrieval via the Open-Meteo historical weather API.
- **Advanced Visualization:** Generate comparative plots for actual vs. predicted temperature trends using matplotlib.
- **Modular and Extensible Design:** Easily expand the project by integrating new models or adapting to additional geographical locations.

---

## Detailed Architectures

1. **GRU with Attention**  
   ```python
   Conv1D → BatchNorm → GRU → Attention → GRU → TimeDistributed Dense
   ```
2. **CNN-GRU Hybrid**  
   ```python
   Conv1D → GRU → RepeatVector → GRU → Dense
   ```
3. **LSTM Encoder-Decoder**  
   ```python
   LSTM → RepeatVector → LSTM → Dense
   ```

---

## **Quick Start Guide**

### 1. Install Dependencies

Use pip to install all required packages:

```bash
pip install openmeteo-requests requests-cache tensorflow pandas numpy matplotlib scikit-learn
```

### 2. Clone the Repository

Clone the repo and change to the project directory:

```bash
git clone https://github.com/yourusername/TempestAI.git
cd TempestAI
```

### 3. Run the Prediction Pipeline

Launch the temperature prediction pipeline by specifying the location and model type:

```bash
python predict_temperature.py --latitude 23.0258 --longitude 72.5873 --model_type GRU_Attention
```

*Example Output:*

![Prediction Plot](https://via.placeholder.com/800x400.png/003366/FFFFFF?text=Actual+vs+Predicted+24h+Forecast)

---

## **Customization Guide**

### Change Location

To modify the forecast location, update the coordinates in `predict_temperature.py`:

```python
# predict_temperature.py
params = {
    "latitude": 40.7128,  # Example: New York City
    "longitude": -74.0060
}
```

### Add a New Model

1. **Extend the `build_model()` Function:**

   Add a new block for your model type. For example, to integrate a Transformer-based model:

   ```python
   elif model_type == "Transformer":
       x = MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
       x = GlobalAveragePooling1D()(x)
       # Further layers can be appended as needed
   ```
2. **Update the Evaluation Suite:**

   Include the new model in the `models` list within `evaluate_all_models()` to compare its performance against existing architectures.

---

## **Performance Metrics**

The following table summarizes the performance metrics for each model based on evaluation datasets:

| **Model**         | **MSE** | **MAE** | **R² Score** |
|-------------------|---------|---------|--------------|
| GRU with Attention| 1.12    | 0.89    | 0.94         |
| LSTM Encoder-Decoder | 1.35 | 1.02    | 0.92         |
| CNN-GRU Hybrid    | 1.28    | 0.97    | 0.93         |
| Dense Model       | 2.01    | 1.31    | 0.87         |

---

## **Future Roadmap**

- [ ] Deploy a real-time API with FastAPI.
- [ ] Integrate ensemble modeling (combining XGBoost with neural networks).
- [ ] Develop an extreme weather event classification module.
- [ ] Enable mobile app integration using Flutter and TensorFlow Lite.

---

## **Contributing**

Contributions are welcome! To propose improvements or new features:

1. **Fork the Repository:**  
2. **Create a Feature Branch:**  
   ```bash
   git checkout -b feature/awesome-model
   ```
3. **Commit Your Changes:**  
   ```bash
   git commit -m "Add Awesome Model"
   ```
4. **Push and Open a Pull Request:**  
   Submit your pull request along with performance metrics and documentation.

---

## **License**

This project is licensed under the MIT License. See [LICENSE.md](LICENSE.md) for details.

---

## **Built With**

[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)]()  
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)]()  
[![Open-Meteo API](https://img.shields.io/badge/Open--Meteo-API-blue)]()

---

## **Contact**

Connect on [LinkedIn](https://linkedin.com/in/shikharpanchal) to discuss ideas, report issues, or contribute to TempestAI.  
*"Have an idea to improve TempestAI? Let's build the future of weather technology together!"*

---

**Predict the Unpredictable.**
