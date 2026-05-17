# NeuroVision AI — AI-Based Detection of Intracranial Aneurysms

**Live Demo:** [Shiko Dashboard-in Këtu](https://blinasopjani.github.io/RSNA-Aneurysm/frontend/dashboard.html)

**Studente:** Blina Sopjani · ID: 69401  
**Universiteti:** Universum · Computer Science  
**Modeli kryesor:** ResNet-101 (Transfer Learning) · AUC: 0.924  

---

## Struktura e Projektit

```
RSNA_Aneurysm_Project/
│
├── backend/                 # API Server (Flask)
│   ├── api.py               # Inference endpoint (ResNet-101)
│   └── .env.example         # Konfigurimi i mjedisit
│
├── frontend/                # Dashboard (Web Application)
│   ├── dashboard.html        # Faqja kryesore
│   ├── css/style.css         # Stilizimi premium
│   └── js/dashboard.js       # Logjika e grafikëve dhe inferimit
│
├── scripts/                 # Skriptat Python
│   ├── preprocessing.py      # Pastrimi dhe normalizimi i të dhënave
│   ├── model_training.py     # Trajnimi i ResNet-101 dhe CNN
│   ├── predict.py            # Parashikimi me modelin e trajnuar
│   ├── hpo_analysis.py       # Hyperparameter Optimization
│   ├── rsna_aneurysm_analysis.py  # Analiza eksploruese (EDA)
│   ├── dataset_connection.py # Lidhja me datasetin RSNA
│   ├── database.py           # Menaxhimi i bazës së të dhënave
│   └── generate_samples.py   # Gjenerimi i imazheve sintetike
│
├── data/                    # Të dhënat
│   ├── train.csv             # Dataset-i origjinal RSNA (4,348 raste)
│   ├── hpo_results.json      # Rezultatet e HPO
│   ├── rsna_schema_and_queries.sql  # Skema SQL
│   └── train_data/           # Imazhet e trajnimit
│
├── samples/                 # Imazhe testimi (pos/neg/anatomical)
│
├── docs/                    # Dokumentacioni
│   ├── Thesis Proposal form Blina.pdf
│   └── Permbledhje_Pune.md   # Përmbledhja e punës
│
├── README.md                # Ky skedar
├── requirements.txt         # Varësitë Python
└── start_project.bat        # Startimi i shpejtë
```

---

## Si të Startohet

```bash
# 1. Instalo varësitë
pip install -r requirements.txt

# 2. Starto projektin (Backend + Frontend)
start_project.bat

# 3. Hap Dashboard-in në browser
http://localhost:5005 → Backend API
dashboard.html     → Frontend (hap direkt në browser)
```

---

## Faqet e Dashboard-it

| Faqja | Përshkrimi |
|-------|------------|
| **Overview** | KPI kryesore: 4,348 serie, 1,863 pozitive, AUC 0.924 |
| **Data Analysis** | Shpërndarja gjinore, moshe, prevalenca sipas modalitetit |
| **Data Cleaning** | Missing values, pastrimi i të dhënave |
| **AI Models** | Tabela krahasuese CNN vs ResNet, ROC Curve, HPO |
| **Training** | Konvergjenca e trajnimit (50 epoka) |
| **Inference Sim** | Upload imazhi → Detektim live me ResNet-101 |
| **Institutions** | Kontributi i 17 qendrave mjekësore |
| **Pipeline** | Procesi i pastrimit: IQR Outlier Detection |

---

## Teknologjitë

- **Python** · Flask, PyTorch, OpenCV, NumPy
- **Deep Learning** · ResNet-101, CNN Baseline
- **Frontend** · HTML5, CSS3, Chart.js, Tabler Icons
- **Dataset** · Global Intracranial Aneurysm Detection Dataset
