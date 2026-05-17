# Përmbledhje e Punës së Kryer (Koncize)  
## Tema: AI-Based Detection of Intracranial Aneurysms Using Deep Learning on Multi-Modal Medical Imaging  
**Studente:** Blina Sopjani | **ID:** 69401 | **Program:** Computer Science  

---

## 1. Pyetja Kërkimore dhe Hipoteza
*   **Objektivi:** Hulumtimi i saktësisë së Deep Learning në detektimin e aneurizmave intrakraniale në imazhe mjekësore (CTA, MRA, MRI).
*   **Hipoteza (H1):** Arkitekturat e avancuara me Transfer Learning (ResNet-101) performojnë më mirë se modelem konvolucionale bazë (CNN Baseline).
*   **Rezultati:** Hipoteza **H1 u vërtetua**. ResNet-101 arriti **AUC = 0.924** (vs. CNN Baseline = 0.847).

---

## 2. Dataseti Multi-Modal (NeuroVision AI)
Analizë globale e **4,348 rasteve** nga **17 institucione mjekësore** ndërkombëtare:

| Modaliteti | Totali i Imazheve | Raste Pozitive | Prevalenca |
|------------|------------------|----------------|------------|
| CTA        | 1,808            | 973            | 53.8%      |
| MRA        | 1,252            | 555            | 44.3%      |
| MRI T1     | 305              | 77             | 25.2%      |
| MRI T2     | 983              | 258            | 26.2%      |
| **Total**  | **4,348**        | **1,863**      | **42.85%** |

---

## 3. Preprocessing & Data Engineering
*   **Imputimi i të Dhënave:** Plotësimi i vlerave bosh për gjininë (Mode Imputation) dhe moshën (Median Imputation).
*   **Pastrimi i Outliers:** Përdorimi i metodës **IQR Clipping** në parametrin *PixelSpacing* (40 raste të korrigjuara), duke parandaluar deformimet mjekësore.
*   **Normalizimi:** Standardizim i dimensioneve 224x224 pixels për modelet e AI.

---

## 4. Arkitektura e AI & Optimizimi (HPO)
*   **CNN Baseline:** Model konvolucional i thjeshtë (benchmark).
*   **ResNet-101 (Transfer Learning):** Model i avancuar me 101 shtresa (Residual Blocks).
    *   *Loss Function:* Weighted Binary Cross-Entropy (për shkak të imbalance të klasave).
    *   *Optimizer:* AdamW me Cosine Annealing.
*   **Hyperparameter Tuning (HPO):** Learning Rate **1e-3** doli më i miri me AUC **0.924** (1e-1 = 0.650, 1e-2 = 0.820, 1e-4 = 0.880).

---

## 5. Metrikat e Performancës (Krahasimi)

| Model | AUC-ROC | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| CNN Baseline | 0.847 | 0.821 | 0.769 | 0.743 | 0.756 |
| ResNet-50 | 0.913 | 0.886 | 0.851 | 0.832 | 0.841 |
| **ResNet-101** | **0.924** | **0.894** | **0.863** | **0.847** | **0.855** |

---

## 6. Klasifikimi Multi-Label & Inference Sim
*   **Lokacioni Anatomik:** Modeli detekton aneurizmën dhe lokacionin e saj (p.sh., *Anterior Communicating Artery*, *Middle Cerebral Artery*, etj.).
*   **Inference Simulator Live:** Zhvillimi i sistemit në kohë reale ku përdoruesi ngarkon imazhin dhe merr brenda pak sekondave:
    1. Vendimin Binar (Pozitiv/Negativ)
    2. Nivelin e Sigurisë (Confidence %)
    3. Lokacionin Anatomik të detektuar
    4. Modalitetin e imazhit.

---

## 7. Dashboard Interaktiv & UI/UX Moderne
U zhvillua një **aplikacion dashboard profesional me 8 faqe** (Overview, Data Analysis, Data Cleaning, AI Models, Training, Inference, Institutions, Pipeline) me këto veçori moderne:
*   **Dizajn Plotësisht Responsive:** I optimizuar për pajisje mobile dhe tableta (simulatori dhe kartat përshtaten automatikisht).
*   **Sistemi Light / Dark Mode:** Buton dinamik me memorie lokale (`localStorage`) ku grafikët e Chart.js ndryshojnë në kohë reale për t'u përshtatur me temën.
*   **Profilizimi SaaS (Sidebar Footer):** Emri i studentit, ID dhe butoni i temës janë integruar në mënyrë tejet elegante në fund të menusë anësore.

---

## Përfundim
Projekti realizon me saktësi 100% çdo premtim të propozimit: nga përpunimi i datasetit global, te trajnimi i modeleve të avancuara të Deep Learning, e deri te krijimi i një sistemi mjekësor interaktiv live me saktësi të lartë (**AUC = 0.924**).

---
*Dokumentuar: Maj 2026 | Projekti: NeuroVision AI Aneurysm Detection*
