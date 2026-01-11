# Quick Start

> Goal: run the inference service locally (Docker / non-Docker) and complete an end-to-end prediction via the HTTP API.

## Option A: Run Locally (Python)

1. Create a virtual environment and install dependencies

<pre class="overflow-visible! px-0!" data-start="485" data-end="622"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>python -m venv .venv
</span><span># Windows: .venv\Scripts\activate</span><span>
</span><span># Linux/Mac: source .venv/bin/activate</span><span>
pip install -r requirements.txt
</span></span></code></div></div></pre>

2. Start the service (different entrypoints correspond to different model combos / strategies)

<pre class="overflow-visible! px-0!" data-start="720" data-end="787"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>python run.py
</span><span># or</span><span>
python run_2.py
</span><span># or</span><span>
python run_3.py
</span></span></code></div></div></pre>

This will start a local HTTP service (port defined in `config.py`) and expose a `/predict` endpoint.

---

## Option B: Run with Docker

1. Build the image

<pre class="overflow-visible! px-0!" data-start="946" data-end="1004"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>docker build -t stroke-predict-engine:latest .
</span></span></code></div></div></pre>

2. Run the container and map the port (replace `<PORT>` with your actual port, e.g., 5000/8000)

<pre class="overflow-visible! px-0!" data-start="1103" data-end="1176"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>docker run --</span><span>rm</span><span> -p <PORT>:<PORT> stroke-predict-engine:latest
</span></span></code></div></div></pre>

---

# API Example

> The inference service accepts a patient-feature JSON payload and returns a stroke prediction (0/1) and optional probability.

## Request (POST /predict)

(Field names depend on your implementation; below is a typical example consistent with the features described in this README.)

<pre class="overflow-visible! px-0!" data-start="1482" data-end="1908"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>curl -X POST http://localhost:<PORT>/predict \
  -H </span><span>"Content-Type: application/json"</span><span> \
  -d '{
    "age": 67,
    "race": 1,
    "sex": 1,
    "sbp": 160,
    "dbp": 95,
    "blood_sugar": 140,
    "htn": 1,
    "dm": 0,
    "hld": 1,
    "smoking": 0,
    "hx_of_stroke": 0,
    "hx_of_afib": 0,
    "hx_of_psych_illness": 0,
    "hx_of_esrd": 0,
    "hx_of_seizure": 0,
    "nihss": 3,
    "facial_droop": 1
  }'
</span></span></code></div></div></pre>

## Response (Example)

<pre class="overflow-visible! px-0!" data-start="1933" data-end="2105"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-json"><span><span>{</span><span>
  </span><span>"prediction"</span><span>:</span><span></span><span>1</span><span>,</span><span>
  </span><span>"probability"</span><span>:</span><span></span><span>0.87</span><span>,</span><span>
  </span><span>"meta"</span><span>:</span><span></span><span>{</span><span>
    </span><span>"model"</span><span>:</span><span></span><span>"xgb | rf | nn | voting"</span><span>,</span><span>
    </span><span>"cluster_id"</span><span>:</span><span></span><span>"optional"</span><span>,</span><span>
    </span><span>"is_boundary"</span><span>:</span><span></span><span>"optional"</span><span>
  </span><span>}</span><span>
</span><span>}</span><span>
</span></span></code></div></div></pre>

---

# Project Structure

> This repository supports both (1) experimentation/training/analysis and (2) deployment-ready inference services. The two tracks are separated but share the same feature processing logic and model artifacts.

<pre class="overflow-visible! px-0!" data-start="2343" data-end="3166"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-text"><span><span>.
├── Dataset - FABS (init).xlsx
├── data_clean.xlsx / data_clean_2.xlsx / data_clean_3.xlsx
├── models/
│   ├── RF_model_for_stroke_prediction.pkl
│   ├── XGB_model_for_stroke_prediction.json
│   └── ML_model_for_stroke_prediction.h5
├── featureset_multimodels/
│   ├── trained_models/
│   └── model_trainer_with_reports*.py
├── model_feature_analysis/
│   └── feature_importance_analyzer.py
├── nearest_neighbor_analyzer/
│   └── nearest_neighbor_analyzer.py
├── processing/
│   ├── interval_divider.py
│   ├── grid_processor.py
│   ├── grid_merger.py
│   ├── gravitational_kmeans.py
│   └── voting_models_1.py
├── app/
│   ├── routes/
│   ├── services/
│   ├── models/
│   ├── templates/
│   ├── static/
│   └── config.py
├── run.py / run_2.py / run_3.py
├── requirements.txt
├── Dockerfile
└── .dockerignore
</span></span></code></div></div></pre>

---

# 0. Goal & Business Context (Ambulance / Pre-hospital Stroke Triage)

 **Goal** : build a lightweight AI inference service that can be deployed on an ambulance (or an edge device / in-vehicle computer):

* Input: **easy-to-collect** pre-hospital patient features (e.g., age, sex, race, SBP/DBP, blood sugar)
* Output: stroke probability / classification (0/1)
* Use cases:  **on-site triage** , early hospital notification, priority ranking, resource allocation

Your key statement:

> “features are the ones that are easy to obtain on an ambulance”

This means the project is not about chasing absolute SOTA accuracy, but about  **deployability, real-time inference, interpretability, and robustness** .

---

# 1. Dataset & Problem Definition (FAB / FABS Stroke Dataset)

### 1.1 Data Overview

* Data: FAB / FABS stroke dataset (also in the repo as `Dataset - FABS (init).xlsx`)
* Sample size: ~ **800**
* Features: ~ **15 features + 1 label**
* Label: 0/1 (your definition: 0=non-stroke, 1=stroke)

> You noted the dataset is imbalanced: majority is 1, minority is 0.
>
> This directly affects evaluation: accuracy can look “fine” while precision/recall/ROC-AUC can be less satisfying, especially for the minority class.

### 1.2 Typical Features (as you described)

* Demographic: age, race, sex
* Vitals: SBP, DBP
* Labs: blood sugar (often measurable pre-hospital)
* And in general: the **pre-hospital accessible feature set**

---

# 2. Data Cleaning & Feature Engineering (From Excel to Trainable Tables)

Your repo includes multiple cleaned versions:

`data_clean.xlsx / data_clean_2.xlsx / data_clean_3.xlsx` — very realistic for iterative experiments.

Core actions:

### 2.1 Data Cleaning

* Remove blank rows/cols, special characters, non-parsable cells
* Normalize categorical encoding (race/sex, etc.)
* Handle missing values (drop / impute / keep missing flags — per your strategy)
* Remove irrelevant columns (IDs / notes / etc.)

### 2.2 Feature Selection

* Keep features that are both **predictive** and **available in pre-hospital settings** (SBP/DBP, etc.)
* Typically guided by:
  * medical prior knowledge (what correlates with stroke)
  * model-driven importance analysis (e.g., `model_feature_analysis/feature_importance_analyzer.py`)

---

# 3. Baseline: Train Three Models Directly (RF / XGBoost / Neural Network)

This is a good “complementary trio”:

* **Random Forest (RF)** : robust, small-data friendly, interpretable via feature importance
* **XGBoost (XGB)** : strong nonlinear generalization on tabular tasks
* **Neural Network (NN)** : expressive, but more sensitive to data size and regularization

Artifacts in the repo:

* `RF_model_for_stroke_prediction.pkl`
* `XGB_model_for_stroke_prediction.json`
* `ML_model_for_stroke_prediction.h5`

### 3.1 Training & Evaluation Metrics (what you mentioned + standard practice)

Likely metrics include:

* Accuracy
* Precision
* Recall / Sensitivity
* F1
* ROC-AUC
* Confusion Matrix (TP/FP/TN/FN)
* (Optional) PR-AUC (often more informative under imbalance)

### 3.2 Baseline Pain Point (your conclusion)

> “Accuracy and precision were not very good”

Common reasons under imbalanced + weak global separability:

* imbalance pushes models toward the majority class (in your case, label=1)
* feature space may contain “multiple local clusters” rather than a single clean boundary
* global training can “mix up” separable local structures

This naturally motivates step 4:  **structure the distribution first, then train** .

---

# 4. Key Innovation: Grid + Clustering + Boundary Expansion + Region-wise Training

This is the project highlight, and the repo structure matches it:

* `processing/grid_processor.py`
* `processing/grid_merger.py`
* `processing/interval_divider.py`
* `processing/gravitational_kmeans.py`
* `processing/voting_models_1.py`
* `nearest_neighbor_analyzer/nearest_neighbor_analyzer.py`

Below is the logic in a clean narrative:

## 4.1 Core Assumptions (important)

1. **Similar samples tend to cluster together** (local density)
2. In many cases clusters are closer to **convex / ball-like** regions than highly concave shapes (k-means-like methods become viable)
3. By discretizing into grids you can **roughly infer the number of clusters and initial centers** (like “data-driven k initialization”)

## 4.2 Grid-based Discovery of Initial Clusters (coarse-to-fine)

Instead of running k-means directly:

* discretize continuous variables into intervals (`interval_divider.py`)
* form multi-dimensional grid cells (`grid_processor.py`)
* compute cell density and label proportions
* under tuned grid rules (your repeated experiments), select:
  * high-density candidate center cells
  * cluster count and initial locations

Value:

**turn the hardest part of k-means (k and initialization) from guesswork into data-driven discovery.**

## 4.3 Boundary Expansion (Boundary Growing / Region Expansion)

You grow regions outward from the initial centers:

* initialize clusters using center cells
* expand boundaries step by step (neighbor cells / radius / threshold)
* at each step:
  * cluster membership changes
  * outside residual set changes (boundary / noise points)

**Added: boundary distance computation & assignment (k-means-like assignment)**

At each boundary expansion step, you apply a k-means-style assignment for “uncertain/boundary” points:

1. define a distance metric in feature space (Euclidean / weighted distance)
2. for each boundary point `x`, compute distances to current cluster centers (or representatives):
   * `d(x, c1), d(x, c2), ..., d(x, ck)`
3. assignment decision:
   * if the closest cluster also satisfies the expansion rule (e.g., distance < r or falls into eligible neighbor cell set), absorb into that cluster
   * otherwise keep in the **boundary / remaining set**

This makes boundary growing a **competitive assignment + stabilization** process (not “hard forcing” points into clusters).

### 4.4 Re-train + Region-wise Evaluation at Every Step

You evaluate clustering by its downstream ML improvement:

* train/evaluate per cluster (or per label-dominant cluster)
* train/evaluate on boundary/remaining as a separate region (fallback)
* combine region models and compare end-to-end metrics

Typical deployed structures:

**Structure A: Cluster-specific models**

* one model per cluster
* inference: locate cluster → run its model

**Structure B: Cluster model + Boundary model**

* cluster models for stable clusters
* boundary fallback model for uncertain points

**Structure C: Ensemble voting**

* `voting_models_1.py` suggests RF/XGB/NN voting or weighted fusion in some regions

## 4.5 Threshold Selection (“best threshold”)

“Threshold” is a bundle of hyperparameters:

* grid granularity, density cutoff, boundary expansion radius, cluster merge threshold, etc.

You choose the threshold that yields best overall ML metrics, achieving:

* Accuracy improved from ~88% → **92–93%**

Interpretation:

> split a globally mixed feature space into locally separable regions, then train region-specific predictors.

**Added: after choosing the best threshold, run 3 enhancement branches and compare**

After selecting the best threshold (clusters + boundary), you do a 3-way comparison:

### Plan A: Threshold-based clustering (baseline threshold result)

* input: clusters + boundary under threshold
* do: region training + boundary fallback (+ optional voting)
* output: end-to-end metrics

### Plan B: Threshold + Gravity-KMeans (re-cluster on top of threshold)

* input: threshold clusters / centers
* do: apply **gravity-kmeans** (`processing/gravitational_kmeans.py`)
  * intuition: “attraction” depends not only on distance but also local density/structure, stabilizing assignments and center updates
* output: new clusters + new boundary; re-train and re-evaluate

### Plan C: Threshold + Gravity-KMeans + Credit-weighted boundary clustering

* input: Plan B results
* do: introduce **credit weights** for boundary points:
  * boundary points receive different credit based on their original label (0/1) or risk direction
  * credit affects effective attraction / distance (some boundary points become easier/harder to absorb)
* output: credit-weighted clusters + boundary; re-train and re-evaluate

### Final selection

Compare A/B/C on:

* overall metrics (Accuracy / Precision / Recall / ROC-AUC / PR-AUC)
* boundary stability & hard cases
* robustness (avoid one bad cluster poisoning the region model)

Pick the best plan as the final clustering + region modeling strategy.

---

# 5. Model Artifacts & Repo Mapping

Clear separation in the repo:

* `models/`: final trained models (.pkl / .json / .h5)
* `featureset_multimodels/trained_models/`: multi-feature-set or multi-threshold variants
* `featureset_multimodels/model_trainer_with_reports*.py`: automated training + reports
* `model_feature_analysis/`: importance & performance analysis
* `nearest_neighbor_analyzer/`: boundary / neighborhood analysis & tuning

This effectively separates:

**experimentation (training/analysis)** vs  **deployment (artifacts + API)** .

---

# 6. Service + Docker (Inference as a REST API)

At the root you have:

* `app/`
  * `routes/`: API routes (`/predict`)
  * `services/`: inference logic (load models, preprocess, predict)
  * `models/model.py`: model wrapper/loading
  * `templates/ static/`: optional web demo
  * `config.py`: ports/paths/env
* `run.py / run_2.py / run_3.py`: different boot modes / model combos
* `requirements.txt`
* `Dockerfile / .dockerignore`

Typical inference flow:

1. API receives JSON (patient features)
2. validate fields/types; missing handling
3. apply the same preprocessing as training (encoding/interval/grid/…)
4. decide: which cluster or boundary
5. call the corresponding model (or voting)
6. return prediction (class + probability + optional meta)

---

# 7. Azure Deployment Workflow (ACR + ACI + External Access)

The end-to-end cloud chain matches your description:

## 7.1 Build Docker image locally

* write `Dockerfile`
* `docker build` to create the image (expose port, e.g., 5000/8000)

## 7.2 Push to Azure Container Registry (ACR)

Conceptual steps:

1. create ACR in Azure
2. login ACR from local
3. tag: `<acr_login_server>/<image_name>:<tag>`
4. push to ACR

You mentioned “container instance + registry” →  **ACR + ACI** .

## 7.3 Run Azure Container Instance (ACI) and expose port

* pull from ACR
* configure:
  * CPU/Memory
  * env vars
  * port exposure
  * DNS label → public FQDN

## 7.4 End-to-end test with Postman

* Postman sends JSON payload
* POST `http://<aci_fqdn>:<port>/predict`
* response:
  * prediction (0/1)
  * probability (if enabled)
  * optional meta (cluster/voting)

---

# 8. One-Line Story for README / Interview / Demo

Suggested main narrative:

1. **Pre-hospital Stroke Prediction** for ambulance-deployable inference
2. **Clean & Select Features** that are available on an ambulance
3. **Baseline Models** : RF/XGB/NN evaluated with Accuracy/Precision/ROC-AUC/CM
4. **Structure-aware Improvement (core contribution)** :

* grid finds cluster count & centers
* boundary expansion + distance assignment stabilizes clusters vs boundary
* post-threshold comparison across:
  * threshold
  * threshold + gravity-kmeans
  * threshold + gravity-kmeans + credit weighting
* metrics improved from ~88% to **92–93%**

1. **Deployment-ready** : artifacts (.pkl/.json/.h5) → REST API → Docker
2. **Azure production-like** : ACR → ACI → Postman end-to-end validation

---

# 9. Azure Cloud Deployment & End-to-End Validation (from Real Portal Screenshots)

After training + Dockerization, the system was deployed and validated on  **Microsoft Azure** . The following details come from actual Azure Portal screenshots, proving a complete cloud inference loop.

---

## 9.1 Resource Group Organization

All cloud resources were placed under a single resource group:

* **Resource Group** : `StrokePredictionResourceGroup`
* Tag:
  * `StrokePredictionTag : StrokePredictionValue`
* Deployments: **16 Succeeded**

This indicates iterative provisioning/adjustment rather than a single one-shot deployment.

---

## 9.2 Azure Container Registry (ACR)

A dedicated ACR was used to manage inference images:

* **ACR Name** : `strokepredictregistry`
* **Login Server** :

<pre class="overflow-visible! px-0!" data-start="15462" data-end="15506"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-text"><span><span>strokepredictregistry.azurecr.io
</span></span></code></div></div></pre>

* **Region** : East US
* **Pricing Tier** : Basic
* **Creation Time** : 2024-11-24 11:04 PM (EST)

Image stored in ACR:

<pre class="overflow-visible! px-0!" data-start="15625" data-end="15698"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-text"><span><span>strokepredictregistry.azurecr.io/stroke-predict-engine:latest
</span></span></code></div></div></pre>

---

## 9.3 Azure Container Instance (ACI)

The inference container was launched from ACR using ACI for cloud validation:

* **ACI Name** : `strokepredictcontainer`
* **Region** : East US
* **OS** : Linux
* **Container Count** : 1
* **Current State** : Stopped (after validation to control cost)
* **Public FQDN** :

<pre class="overflow-visible! px-0!" data-start="16010" data-end="16082"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-text"><span><span>strokepredictengine-dzhbfh3epg2dhgu.eastus.azurecontainer.io
</span></span></code></div></div></pre>

The ACI resource also shows CPU/Memory/Network metrics, indicating it was actually run and served traffic.

---

## 9.4 Azure App Service (Container-based) as an Alternative Hosting Mode

In addition to ACI, the same image was deployed via **Azure App Service for Containers** to compare hosting options:

* **App Service Name** : `strokepredictengine`
* **OS** : Linux
* **Region** : Canada Central
* **App Service Plan** : `ASP-StrokePredictionResourceGroup-91e0`
* Tier: Basic (B1)
* **Publishing Model** : Container
* **Image Source** : Azure Container Registry
* **Image** :

<pre class="overflow-visible! px-0!" data-start="16659" data-end="16732"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-text"><span><span>strokepredictregistry.azurecr.io/stroke-predict-engine:latest
</span></span></code></div></div></pre>

---

## 9.5 Multi-region Deployment Footprint (Experiment Evidence)

From the portal:

* **ACR / ACI** : East US
* **App Service / Plan** : Canada Central

This suggests region selection was tested (availability/cost/experiment needs).

---

## 9.6 End-to-End Cloud Inference Test (Postman)

After the container was running:

* Send HTTP POST requests to public FQDN + exposed port
* Input JSON includes ambulance-accessible fields (age/race/sex/SBP/DBP/blood sugar/…)
* Output includes prediction (0/1), probability (optional), and optional cluster/voting meta

This confirms:  **cloud container → model load → preprocessing → inference → response** .

---

## 9.7 Azure Subscription Context

* **Subscription** : Microsoft Azure Sponsorship
* Status: Active
* Resource counts (under this subscription):
  * Container Registries: 1
  * Container Instances: 1
  * App Services: 1
  * App Service Plans: 1

---

## 9.8 Summary

Azure evidence confirms:

* Dockerized inference service
* Image pushed to ACR
* Service successfully ran on ACI with a public endpoint
* Also validated with App Service (container-based)
* End-to-end call verified via Postman

This project is therefore a  **full end-to-end stroke prediction prototype with real cloud deployment and validation** , not just a local ML experiment.

---

# 10. Feature Importance & Feature-set Combination Analysis (Models + SHAP + Deployability Constraints)

Before structure-aware clustering, the project performed systematic analysis of **feature importance and feature-set combinations** using multiple tools:  **model-based importance, SHAP explainability, ablation tests, and real deployability constraints** .

---

## 10.1 Input Specification & Real-world Constraints

The project clarified feasible input constraints:

* Age: 1–150
* Race: 1=C, 2=AA, 3=Other (often merged as Other)
* Sex: 1=Male, 2=Female
* Binary history features (0/1):
  * HTN, DM, HLD, Smoking
  * HxOfStroke, HxOfAfib, HxOfPsychIllness
  * HxOfESRD, HxOfSeizure
  * FacialDroop (weakness)
* Continuous vitals/labs:
  * SBP / DBP (mmHg)
  * BloodSugar (mg/dL)
* NIHSS score

Workflow note:

* **NIHSS and FacialDroop** may come from teleneurologist assessment and may not be reliably available on-scene.
* This motivates:
  > “What happens if we remove these features?”
  >

---

## 10.2 Multi-model Feature Importance (DT / RF / LightGBM)

### 10.2.1 Models and shared hyperparameters

Importance analysis was run on:

* Decision Tree (DT)
* Random Forest (RF)
* LightGBM (GBM)

With key shared settings:

* `max_depth = 6`
* `min_samples_leaf = 20`
* `n_estimators = 100` (RF / LightGBM)

### 10.2.2 Cross-model consistency

Under the full feature set:

* **FacialDroop, SBP, Age, BloodSugar** consistently ranked high across DT/RF/LightGBM
* Different models emphasized different patterns:
  * DT/RF: strong individual features
  * LightGBM: more distributed contributions

Key takeaway:

> Only features that are important **across multiple models** are treated as stable core candidates.

---

## 10.3 Correlation & Redundancy Analysis

A feature correlation heatmap was used to:

* detect redundancy (highly correlated features)
* keep the feature space “cleaner” for clustering/grid partitioning
* distinguish independent risk signals vs proxies

---

## 10.4 SHAP: From “Important” to “Explainable”

SHAP was used on DT/RF/LightGBM to understand:

* whether a feature consistently increases/decreases stroke risk
* whether the direction is consistent across models
* whether effects are global or only for a few samples

### 10.4.2 SHAP distribution interpretation

Observations typically include:

* **SBP/DBP** show stable SHAP distributions:
  * high values → higher predicted risk
  * low values → lower risk
* noisy features show mixed positive/negative contributions

Key conclusion:

> SBP/DBP are not only “important” but also  **behaviorally consistent and medically plausible** .

---

## 10.5 Cross-validated Importance Stability

RF/LightGBM importance rankings were checked across different splits:

* SBP, Age, BloodSugar remain stable
* weaker features fluctuate more with splits

This increases confidence in the selected core set.

---

## 10.6 Ablation: Removing NIHSS & FacialDroop

Given pre-hospital constraints:

### 10.6.1 Design

Compare:

1. Full feature set (with NIHSS + FacialDroop)
2. Reduced feature set (without NIHSS + FacialDroop)

Evaluate DT/RF/LightGBM using:

* Accuracy, Precision, Recall, F1, AUC

### 10.6.2 Outcome

* Removing NIHSS/FacialDroop reduces performance (AUC/Accuracy drop)
* But core features remain strong:
  * **SBP, DBP, Age, BloodSugar** stay among the most important
  * model retains meaningful discrimination

Engineering conclusion:

> NIHSS/FacialDroop are powerful but not always obtainable pre-hospital;
>
> SBP/DBP balance **availability + importance + stability** best.

---

## 10.7 Summary: Feature Selection Decision Logic

Final feature selection followed this chain:

1. multi-model importance (DT/RF/LightGBM)
2. SHAP global + local explainability
3. correlation & redundancy check
4. cross-validated stability
5. ablation and feature-set comparisons
6. real-world ambulance workflow constraints

Result:  **SBP, DBP, Age, BloodSugar** , etc., were confirmed as:

> stable, explainable, deployable core features suitable for structure-aware clustering and cloud inference.
