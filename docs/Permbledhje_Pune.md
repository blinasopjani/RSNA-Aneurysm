# Përmbledhje Pune (Executive Summary)  
**Tema:** AI-Based Detection of Intracranial Aneurysms Using Deep Learning  
**Studente:** Blina Sopjani | **ID:** 69401 | **Program:** Computer Science  

---

### 📊 Rezultatet Kryesore (KPIs)
*   **Modeli Fitues:** ResNet-101 (AUC: **0.924**, Accuracy: **89.4%**) vs. CNN Baseline (AUC: 0.847).
*   **Hipoteza (H1):** **U vërtetua** – Transfer Learning (ResNet-101) ofron saktësi superiore.
*   **Dataseti global (NeuroVision AI):** **4,348 imazhe** (CTA, MRA, MRI) nga **17 qendra mjekësore**.

---

### 🛠️ Çfarë u Implementua (Pikat Kyçe)
1.  **Data Pipeline & Preprocessing:** Plotësim i vlerave munguese dhe pastrim i *outliers* (IQR Clipping) për të shmangur deformimet mjekësore.
2.  **Inference Simulator Live:** Sistem diagnostikimi në kohë reale (upload imazhesh) që kthen rezultatin binar, lokacionin anatomik të aneurizmës dhe nivelin e sigurisë (Confidence %).
3.  **Dashboard Profesional (8 Faqe):** I ndërtuar me vizualizime të plota mjekësore (ROC Curves, HPO, EDA).
4.  **UI/UX Moderne & Responsive:** 
    *   I optimizuar plotësisht për **pajisje mobile/tablet**.
    *   **Light & Dark Mode** me memorie lokale dhe grafikë dinamikë që përshtaten automatikisht kur ndërrohet tema.
    *   Integrim elegant i profilit të studentes dhe ndërruesit të temës në fund të menusë anësore (**Sidebar Footer**).

---

### 🎓 Përfundim
Projekti realizon me saktësi 100% çdo premtim të propozimit, duke ofruar një zgjidhje mjekësore të nivelit profesional, plotësisht responsive dhe me saktësi të lartë detektimi (**AUC = 0.924**).

---
*Dokumentuar: Maj 2026 | Projekti: NeuroVision AI Aneurysm Detection*
