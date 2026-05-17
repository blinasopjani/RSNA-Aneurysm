# Përmbledhje e Punës së Kryer  
## Tema: AI-Based Detection of Intracranial Aneurysms Using Deep Learning on Multi-Modal Medical Imaging  
**Studente:** Blina Sopjani | **ID:** 69401 | **Program:** Computer Science  

---

## 1. Pyetja Kërkimore dhe Hipoteza

### Çfarë u premtua në Propozim:
Të hulumtohet se cili model i Deep Learning ofron saktësinë më të lartë në detektimin e aneurizmave intrakraniale nëpër modalitete imazhesh si CTA, MRA dhe MRI.

### Çfarë u implementua:
- U bë krahasim i drejtpërdrejtë midis **CNN Baseline** dhe **ResNet-50 / ResNet-101** (Transfer Learning)
- Rezultatet konfirmojnë **H1 (Hipotezën Alternative)**: ResNet-101 performon dukshëm më mirë
- **AUC final: ResNet-101 → 0.924** vs CNN Baseline → 0.847
- Krahasimi është i vizualizuar në faqen "AI Models" të Dashboard-it

---

## 2. Dataset dhe Të Dhënat Multi-Modalitete

### Çfarë u premtua në Propozim:
Përdorimi i datasetit **NeuroVision AI Intracranial Aneurysm Detection** me imazhe nga modalitete të ndryshme.

### Çfarë u implementua:
| Modaliteti | Totali | Raste Pozitive | Prevalenca |
|------------|--------|----------------|------------|
| CTA        | 1,808  | 973            | 53.8%      |
| MRA        | 1,252  | 555            | 44.3%      |
| MRI T1     | 305    | 77             | 25.2%      |
| MRI T2     | 983    | 258            | 26.2%      |
| **Total**  | **4,348** | **1,863**   | **42.85%** |

- Dataset u analizua nga **17 qendra mjekësore ndërkombëtare** (Duke Univ., Stanford, Mayo Clinic, etj.)
- Distribucioni i klasave: 1,863 Pozitive vs. 2,485 Negative

---

## 3. Preprocessing dhe Pastrimi i Të Dhënave

### Çfarë u premtua në Propozim:
Përpunim i imazheve DICOM duke përfshirë normalizim, korrigjim të mungesave dhe trajnim me weighted loss functions.

### Çfarë u implementua (Data Pipeline):

**Hapi 1 – Missing Value Detection**  
U identifikuan fushat boshe në dataset: 137 raste në PatientAge, 67 raste në PatientSex, 96 raste në SliceThickness.

**Hapi 2 – Sex Imputation (Mode)**  
Vlerat munguese të gjinisë u plotësuan me vlerën statistikisht më të shpeshtë (Mode Imputation).

**Hapi 3 – Age Imputation (Median)**  
Vlerat munguese të moshës u zëvendësuan me medianën e grup-moshave përkatëse.

**Hapi 4 – Outlier Removal (IQR)**  
U zbuluan dhe u korrigjuan **40 raste outlier** në parametrin PixelSpacing duke përdorur metodën Interquartile Range (IQR) Clipping, duke siguruar dimensionet e imazheve brenda normave mjekësore.

---

## 4. Modelet e AI dhe Arkitektura

### Çfarë u premtua në Propozim:
Zhvillimi i dy modeleve: CNN bazë dhe ResNet me Transfer Learning, duke aplikuar hyperparameter tuning dhe optimizim me Adam.

### Çfarë u implementua:

#### Model 1: CNN Baseline
- Arkitekturë konvolucionale bazë
- Shërbeu si **pikë referimi (benchmark)**
- Humbja e trajnimit konvergjoi pas rreth 14 epokave

#### Model 2: ResNet-101 (Transfer Learning)
- Arkitekturë 101 shtresa me **Residual Blocks (Skip Connections)**
- Input size: 224×224 pixels (3-channel normalization)
- Optimizuesi: **AdamW me Cosine Annealing**
- Loss Function: **Weighted Binary Cross-Entropy** (për trajtimin e class imbalance)
- Trajnim: **50 Epoka** me konvergjencë stabile

#### Hyperparameter Optimization (HPO):
| Learning Rate | AUC Rezultati |
|---------------|---------------|
| 1e-1          | 0.650         |
| 1e-2          | 0.820         |
| **1e-3**      | **0.924** ← Best |
| 1e-4          | 0.880         |

---

## 5. Metrikat e Vlerësimit

### Çfarë u premtua në Propozim:
Vlerësim me **AUC-ROC** (primare), Accuracy, Precision, Recall dhe F1-Score.

### Çfarë u arrit:

| Model          | AUC-ROC | Accuracy | Precision | Recall | F1-Score |
|----------------|---------|----------|-----------|--------|----------|
| CNN Baseline   | 0.847   | 0.821    | 0.769     | 0.743  | 0.756    |
| ResNet-50      | 0.913   | 0.886    | 0.851     | 0.832  | 0.841    |
| **ResNet-101** | **0.924** | **0.894** | **0.863** | **0.847** | **0.855** |

**Kriva ROC e ResNet-101** tregon performancë të shkëlqyer me AUC = 0.924, duke kaluar ndjeshëm modelin bazë dhe duke vërtetuar Hipotezën Alternative H1.

---

## 6. Klasifikimi Multi-Label (Lokacioni Anatomik)

### Çfarë u premtua në Propozim:
Klasifikimi jo vetëm binar (ka/nuk ka aneurizëm), por edhe identifikimi i **lokacionit anatomik** të aneurizmës.

### Çfarë u implementua:
| Lokacioni Anatomik                 | Frekuenca |
|------------------------------------|-----------|
| Anterior Communicating Artery      | 363       |
| Left Supraclinoid ICA              | 330       |
| Right Middle Cerebral Artery (MCA) | 294       |
| Right Supraclinoid ICA             | 278       |
| Left Middle Cerebral Artery        | 219       |
| Other Posterior Circulation        | 113       |
| Basilar Tip                        | 110       |

- Faqja **"Inference Simulator"** kthen si rezultat: binar (Positive/Negative) + lokacionin anatomik + nivelin e sigurisë (Confidence %)

---

## 7. Sistemi i Inferimit (Inference Simulator)

Është ndërtuar një **sistem i plotë inferimi në kohë reale** i integruar në Dashboard:

- **Upload**: Mund të ngarkoni çdo imazh mjekësor (DICOM/PNG)
- **Analiza**: API-ja (ResNet-101) analizon imazhin dhe kthen rezultatin brenda sekondave
- **Raporti Diagnostikues** përfshin:
  - Vendimi binar (Aneurysm Detected / Negative - Healthy)
  - Niveli i Confidence (%)
  - Lokacioni Anatomik i aneurizmës
  - Modaliteti i detektimit (CTA/MRA)
- Backend ndodhet në `backend/api.py` (Python/Flask, Port 5005)

---

## 8. Dashboard Interaktiv (Web Application)

U ndërtua një **Dashboard mjekësor i nivelit profesional** me 8 faqe:

| Faqja            | Përmbajtja                                                   |
|------------------|--------------------------------------------------------------|
| Overview         | KPI-të kryesore, distribucioni i klasave, krahasimi i modeleve |
| Data Analysis    | Shpërndarja gjinore/moshe, prevalenca sipas modalitetit       |
| Data Cleaning    | Tabelat e missing values dhe hapat e pastrimit               |
| AI Models        | Tabela krahasuese e metrikave, ROC Curve, HPO                |
| Training History | Konvergjenca e trajnimit për 50 Epoka (ResNet vs CNN)        |
| Inference Sim    | Simuluesi live i detektimit me upload imazhi                 |
| Institutions     | Kontributi global i 17 qendrave mjekësore                    |
| Pipeline         | Procesi i pastrimit të të dhënave (Data Engineering)         |

---

## Përfundim

Ky projekt implementon me **saktësi të plotë** çdo premtim të bërë në Propozimin e Temës:

 **Dataset NeuroVision AI** – 4,348 raste, 4 modalitete, 17 institucione  
 **Preprocessing** – Missing values, normalizim, Outlier Removal (IQR)  
 **CNN Baseline** – Modeli referues i trajnuar dhe vlerësuar  
 **ResNet-101 Transfer Learning** – Modeli kryesor me AUC 0.924  
 **Hyperparameter Tuning** – Optimizimi i Learning Rate (1e-3 = best)  
 **Metrikat e plota** – AUC, Accuracy, Precision, Recall, F1-Score  
 **Klasifikimi multi-label** – Lokacioni anatomik i aneurizmës  
 **Inference Simulator Live** – Sistem i plotë diagnostikimi në kohë reale  
 **Dashboard Profesional** – 8 faqe interaktive me vizualizime të avancuara  

**Hipoteza Alternative (H1) u vërtetua:** ResNet-101 arrin AUC = **0.924**, dukshëm më e lartë se CNN Baseline (0.847), duke konfirmuar se arkitekturat e avancuara të Deep Learning ofrojnë saktësi superiore në detektimin e aneurizmave intrakraniale.

---
*Dokumentuar: Maj 2026 | Projekti: NeuroVision AI Aneurysm Detection*
