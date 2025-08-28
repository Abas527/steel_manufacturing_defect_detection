# 🏭 Manufacturing Defect Detection

An end-to-end **deep learning project** for detecting defects in steel manufacturing.
This project demonstrates **MLOps best practices** using **PyTorch, DVC, MLflow, Streamlit, and Docker**.

---

## 📌 Features

* 🔄 **Data versioning** with **DVC**
* 🧠 **Deep Learning (CNN)** for defect detection
* 📊 **Experiment tracking** with **MLflow & DagsHub**
* 🚀 **CI/CD pipeline** for reproducibility and deployment
* 🌐 **Streamlit Web App** for model inference
* 📦 **Dockerized setup** for portability

---

## 📂 Project Structure

```bash
manufacturing_defect_detection/
│── data/                 # Raw and processed datasets (DVC tracked)
│── src/                  # Source code
│   ├── data_loader.py    # Data preprocessing & loading
│   ├── classification/
│   │   └── image_classification.py  # Model training script
│   ├── evaluate.py       # Model evaluation
│── dvc.yaml              # DVC pipeline stages
│── params.yaml           # Hyperparameters
│── requirements.txt      # Python dependencies
│── Dockerfile            # Docker setup
│── README.md             # Project documentation
│── model.pth             # Trained model (DVC tracked)
```

---

## ⚙️ Pipeline (DVC)

The project uses **DVC** to manage the pipeline:

```yaml
stages:
  prepare_data:
    cmd: python src/data_loader.py
    outs:
      - data/processed

  train:
    cmd: python src/classification/image_classification.py
    deps:
      - data/processed
    outs:
      - model.pth

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - model.pth
      - data/processed
    metrics:
      - metrics.json
```

---

## 🚀 Getting Started

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Abas527/manufacturing_defect_detection.git
cd manufacturing_defect_detection
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Reproduce pipeline

```bash
dvc repro
```

### 5️⃣ Track experiments with MLflow

```bash
mlflow ui
```

Go to **[http://127.0.0.1:5000](http://127.0.0.1:5000)** to view experiment logs.

---

## 🌐 Run Streamlit App

```bash
streamlit run app.py
```

---

## 📦 Run with Docker

```bash
docker build -t defect-detection .
docker run -p 8501:8501 defect-detection
```

---

## 📊 Experiment Tracking with DagsHub

MLflow is integrated with **DagsHub** for remote tracking.
Update your script with:

```python
import dagshub
dagshub.init(repo_owner="your-username", repo_name="manufacturing_defect_detection", mlflow=True)
```

---

## 📈 Results

* Trained CNN achieves **98% accuracy** on validation set.
* Metrics are stored in `metrics.json` and logged in MLflow.

---

## 🔮 Future Improvements

* ✅ Add hyperparameter optimization (Optuna)
* ✅ Deploy model via FastAPI + Streamlit
* ✅ Enhance data augmentation for better generalization

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss.

---

## 📜 License

This project is licensed under the **MIT License**.
