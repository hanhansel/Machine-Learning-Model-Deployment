<a id="top"></a>

<div align="center">

# **Hotel Booking Cancellation Prediction**

<b><ins>An end-to-end machine learning project that predicts whether a hotel booking will be canceled, from data cleaning and model comparison to a Streamlit web app.</ins></b>

<br>

![Status](https://img.shields.io/badge/status-completed-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

</div>

<br>

<details open>
<summary><b>Table of Contents</b></summary>

1. [Project Overview](#project-overview)
2. [Tech Stack](#tech-stack)
3. [How It Works](#how-it-works)
4. [Skills Learned](#skills-learned)
5. [Challenges and Solutions](#challenges-and-solutions)
6. [Project Structure](#project-structure)
7. [Getting Started](#getting-started)
8. [Extra Things](#extra-things)
9. [Credits and Contact](#credits-and-contact)

</details>

---

## <ins>Project Overview</ins>

> **TL;DR:** I trained and compared **Random Forest** and **XGBoost** on **36,275 hotel bookings**. Random Forest won with **90.2% accuracy**, and I deployed it as a **Streamlit** app that predicts whether a booking will be canceled.

### ▸ Problem
Cancellations make it hard for hotels to plan rooms, staff, and revenue. If a hotel knows which bookings are likely to be canceled, it can act early, for example by overbooking carefully or following up with the guest. The goal of this project is to predict, from the details of a booking, whether it will be Canceled or Not Canceled, and to make that prediction available through a simple web app.

### ▸ Objectives
- *Clean the booking data and handle its missing values* with a clear reason for each choice.
- *Train and compare two models*, Random Forest and XGBoost, and keep the better one.
- *Save the model together with its encoders* so the app preprocesses inputs exactly like training did.
- *Deploy the model as a Streamlit app*, with a script for predicting a whole CSV file at once.

### ▸ Context
Built for my Model Deployment midterm exam at Bina Nusantara University. The task was to build a full machine learning pipeline, compare Random Forest and XGBoost, save the best model with pickle, and deploy it with Streamlit.

<p align="right">(<a href="#top">back to top</a>)</p>

---

## <ins>Tech Stack</ins>

> The tools I **used** and what I used them **for**.

| Category | Tool | What I Used It For |
|:---|:---|:---|
| **Language** | `Python` | The training notebook, the model class, the web app, and the batch inference script |
| **Framework** | `Streamlit` | The web app to retrieve input for booking details and show the final prediction result |
| **Data Processing** | `pandas`, `NumPy` | Loading the dataset and preprocessing data |
| **AI / ML** | `scikit-learn`, `XGBoost` | One-hot and label encoding, the train/test split, training Random Forest and XGBoost, and evaluating performance |
| **Visualization** | `matplotlib` | Plots for EDA and preprocessing data |
| **Libraries** | `joblib` | Saving and loading the model and its encoders as one file |
| **Tools** | Jupyter, `Git` | Training experiments and version control |

<p align="right">(<a href="#top">back to top</a>)</p>

---

## <ins>How It Works</ins>

> Clean the data → train two models → save the winner with its encoders → serve it through a Streamlit form or a batch script.

```
Booking Data → Cleaning → Encoding → Train RF + XGBoost → Save Best Model (.pkl) → Streamlit App / Batch Inference
```

1. **Clean.** The dataset has 36,275 bookings. Three columns have missing values (2.5–3.5%). *type_of_meal_plan* is filled with the most common value. *avg_price_per_room* and *required_car_parking_space* are skewed, so they're filled with the median.
2. **Encode and split.** *Booking_ID* is dropped. The data is split 80/20 into training and test sets. The three categorical features (*meal plan*, *room type*, *market segment*) are one-hot encoded, so the model doesn't assume an order between categories. The target (*Canceled* / *Not_Canceled*) is label-encoded.
3. **Train and compare.** Random Forest and XGBoost are trained on the same data. Random Forest scores higher on accuracy and on every metric for the *Canceled* class, so it's kept.
4. **Save.** The Random Forest model, the one-hot encoder, and the label encoder are saved together in `best_model.pkl` with `joblib`.
5. **Serve.**
   - `hotel_booking_model.py` wraps everything in a `HotelBookingModel` class that loads the file, applies the same encoding, and returns *Canceled* or *Not_Canceled*.
   - `app.py` is a Streamlit form with 17 booking inputs. The slider ranges come from the minimum and maximum values of each column in the dataset.
   - `inference.py` predicts a whole CSV of new bookings and saves the results to `predictions.csv`.

<p align="right">(<a href="#top">back to top</a>)</p>

---

## <ins>Skills Learned</ins>

> What this project taught me, technically and beyond.

- **Building an end-to-end ML pipeline:** Taking a project from raw data through cleaning, training and evaluation to a working app. See `code` and `app.py`.
- **Handling missing values:** Checking each column's distribution with box plots first, then using the median for skewed numbers and the mode for categories. See `code`.
- **Encoding categorical data:** Using one-hot encoding for features with no natural order and label encoding for the target. See `code`.
- **Model comparison and evaluation:** Comparing models on accuracy and on per-class precision, recall and F1, not just one number. See `code`.
- **Model serialization:** Saving the model and its encoders as one file, so training and serving always use the same preprocessing. See `code` and `best_model.pkl`.
- **Object-oriented code for ML:** Wrapping loading, preprocessing and prediction in one class that both the app and the batch script reuse. See `hotel_booking_model.py`.
- **Deploying with Streamlit:** Building a form-based web app around a trained model. See `app.py`.
- **Batch inference:** Writing a script that predicts a whole CSV file and saves the results. See `inference.py`.

<p align="right">(<a href="#top">back to top</a>)</p>

---

## <ins>Challenges and Solutions</ins>

> What went wrong, how I fixed it, and what I took away from the project.

**Picking between two close models.** Random Forest reached 90.2% accuracy and XGBoost 88.9%, which is a small gap. To decide properly, I looked at the *Canceled* class, which is the one that matters for the hotel and makes up only a third of the bookings. Random Forest was better there on precision, recall and F1, so the choice wasn't based on accuracy alone.

**Keeping training and the app consistent.** A model only works if new inputs are encoded exactly like the training data. I saved the one-hot encoder and label encoder in the same file as the model and put the preprocessing inside the `HotelBookingModel` class, so any code that loads the model reuses the exact encoders from training instead of rebuilding them.

<p align="right">(<a href="#top">back to top</a>)</p>

---

## <ins>Project Structure</ins>

```
Machine-Learning-Model-Deployment/
├── code                      # training notebook (Jupyter): cleaning, training, comparison
├── best_model.pkl            # Random Forest + one-hot encoder + label encoder
├── hotel_booking_model.py    # HotelBookingModel class: load, preprocess, predict
├── app.py                    # Streamlit web app
├── inference.py              # batch predictions from a CSV file
├── requirements.txt
└── README.md
```

<p align="right">(<a href="#top">back to top</a>)</p>

---

## <ins>Getting Started</ins>

```bash
# 1. Clone
git clone https://github.com/hanhansel/Machine-Learning-Model-Deployment.git
cd Machine-Learning-Model-Deployment

# 2. Install
pip install -r requirements.txt

# 3. Run the web app
streamlit run app.py

# Or predict a CSV of new bookings (saves predictions.csv)
python inference.py
```

> **Requirements:** Python 3 and the packages in `requirements.txt`. `best_model.pkl` was saved with a specific scikit-learn version, so if loading fails, install the version it was saved with. For `inference.py`, put your bookings in `new_data.csv` with the same 17 columns as the app.

<p align="right">(<a href="#top">back to top</a>)</p>

---

## <ins>Extra Things</ins>

<details>
<summary><b>Dataset</b></summary>
<br>

- **Source:** `Dataset_B_hotel.csv`, provided by the university course
- **Size:** 36,275 bookings × 19 columns, with arrivals in 2017 and 2018
- **Target:** *booking_status*: *Canceled* or *Not_Canceled*
- **Features:** number of adults and children, weekend and week nights, meal plan, parking, room type, lead time, arrival date, market segment, repeated guest, previous cancellations and bookings, average room price, and special requests
- **Preprocessing:** mode imputation for *type_of_meal_plan*; median imputation for *avg_price_per_room* and *required_car_parking_space*; one-hot encoding for the three categorical features; label encoding for the target

</details>

<details>
<summary><b>Results and Key Findings</b></summary>
<br>

> **Headline:** **Random Forest** reached **90.2% accuracy** on the test set (7,255 bookings), ahead of XGBoost at 88.9%.

| Model | Accuracy | Canceled Precision | Canceled Recall | Canceled F1 |
|:---|:---:|:---:|:---:|:---:|
| XGBoost | 88.9% | 0.86 | 0.79 | 0.83 |
| **Random Forest** | **90.2%** | **0.88** | **0.82** | **0.85** |

1. **Random Forest wins on every metric for cancellations:** that's the class the hotel cares most about.
2. **Cancellations are harder to predict:** both models recall about 94% of *Not_Canceled* bookings but only about 80% of *Canceled* ones, partly because cancellations are the minority class.
3. **XGBoost shows a small gap between training (91.8%) and test (88.9%) accuracy,** so it generalizes slightly worse on this data with default settings.

</details>

<details>
<summary><b>Model Details</b></summary>
<br>

- **Model:** `RandomForestClassifier` with default settings, `random_state=42`
- **Input:** 17 booking features → 30 model features after one-hot encoding (14 numeric + 16 one-hot)
- **Output:** *Canceled* or *Not_Canceled*
- **Saved as:** `best_model.pkl`

</details>

<p align="right">(<a href="#top">back to top</a>)</p>

---

## <ins>Credits and Contact</ins>

**Acknowledgements:** Dataset provided as part of the Model Deployment course at Bina Nusantara University.

**License:** Personal use only.

**Author:** **Hansel Sebastian**, *Data Science Student @ Bina Nusantara University*

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/hanhansel)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/hansel-sebastian-5a19b8245)

<p align="right">(<a href="#top">back to top</a>)</p>
