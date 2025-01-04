# ProgettoTeam2 (ITA)

## Introduzione

Il presente lavoro fa parte del progetto assegnato al Team 2 del gruppo Data3 2024-2025 nel contesto del corso di data analisi di DevelHope. 

Il progetto consiste nell'analisi dei dati medici di 2139 pazienti affetti da HIV e nella creazione di modelli di machine learning in grado di predire la variabile binaria di target "infected" che segnala lo sviluppo della malattia in AIDS. 

## Il dataset

**I dati:** sono presi da Kaggle in formato csv https://www.kaggle.com/datasets/aadarshvelu/aids-virus-infection-prediction

**La provenienza:** il dataset originale proviene da uno studio del 1995 in cui circa 2100 pazienti affetti da HIV vengono trattati con vari farmaci antiretrovirali https://clinicaltrials.gov/study/NCT00000625
`A Randomized, Double-Blind Phase II/III Trial of Monotherapy vs. Combination Therapy With Nucleoside Analogs in HIV-Infected Persons With CD4 Cells of 200-500/mm3`

**Una pubblicazione:** che usa e spiega questo dataset: https://www.nejm.org/doi/10.1056/NEJM199610103351502?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200www.ncbi.nlm.nih.gov

## Copiando dalla risorsa Kaggle di origine:

### Context :
Dataset contains healthcare statistics and categorical information about patients who have been diagnosed with AIDS. This dataset was initially published in 1996.

### Attribute Information :

- time: time to failure or censoring
- trt: treatment indicator (0 = ZDV only; 1 = ZDV + ddI, 2 = ZDV + Zal, 3 = ddI only)
- age: age (yrs) at baseline
- wtkg: weight (kg) at baseline
- hemo: hemophilia (0=no, 1=yes)
- homo: homosexual activity (0=no, 1=yes)
- drugs: history of IV drug use (0=no, 1=yes)
- karnof: Karnofsky score (on a scale of 0-100)
- oprior: Non-ZDV antiretroviral therapy pre-175 (0=no, 1=yes)
- z30: ZDV in the 30 days prior to 175 (0=no, 1=yes)
- preanti: days pre-175 anti-retroviral therapy
- race: race (0=White, 1=non-white)
- gender: gender (0=F, 1=M)
- str2: antiretroviral history (0=naive, 1=experienced)
- strat: antiretroviral history stratification (1='Antiretroviral Naive',2='> 1 but <= 52 weeks of prior antiretroviral therapy',3='> 52 weeks)
- symptom: symptomatic indicator (0=asymp, 1=symp)
- treat: treatment indicator (0=ZDV only, 1=others)
- offtrt: indicator of off-trt before 96+/-5 weeks (0=no,1=yes)
- cd40: CD4 at baseline
- cd420: CD4 at 20+/-5 weeks
- cd80: CD8 at baseline
- cd820: CD8 at 20+/-5 weeks
- infected: is infected with AIDS (0=No, 1=Yes)

### Additional Variable Information :

- Personal information (age, weight, race, gender, sexual activity)
- Medical history (hemophilia, history of IV drugs)
- Treatment history (ZDV/non-ZDV treatment history)
- Lab results (CD4/CD8 counts)