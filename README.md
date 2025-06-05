##  Segformer Semantic Segmentation API

A FastAPI-based web service for performing image segmentation using [NVIDIA's Segformer](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512) model.

### 🔧 Features

* Accepts image uploads via `POST`
* Returns segmented output image (PNG format)
* Built using:

  * FastAPI
  * Transformers (HuggingFace)
  * PyTorch
  * Uvicorn

---

###  How to Run Locally

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/segformer-api.git
   cd segformer-api
   ```

2. **Create virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Start the server**

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 10000
   ```

---

###  API Usage

**Endpoint:**
`POST /segment/`

**Request:**

* Content-Type: `multipart/form-data`
* Body: Image file (`file` field)

**Response:**

* Segmented PNG image


---

### ☁️ Deployment on Render

> This app is ready for deployment on [Render](https://render.com).

Render auto-builds the service using the included `render.yaml`.

---
