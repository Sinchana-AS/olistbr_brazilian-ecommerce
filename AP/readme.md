# 🚢 Maersk E-commerce AI Assistant

A **GenAI-based agentic system** for querying and analyzing Brazilian e-commerce data using natural language.  
This system lets users chat with structured datasets, uncover insights, and visualize patterns interactively.

---

## 🎥 Demo Video
[📺 Watch the 6-minute demo here](https://drive.google.com/file/d/1XmjH8Fsj2ydDVDoUY-STGglh-C6ww3uo/view?usp=drive_link)

## 💻 GitHub Repository
[🔗 GitHub Repo](https://github.com/Sinchana-AS/olistbr_brazilian-ecommerce)

---

## 🎯 Features

### Core Capabilities
- 💬 **Natural Language Queries** – Ask questions in plain English
- 📊 **Smart Data Analysis** – SQL generation from natural language
- 📈 **Interactive Visualizations** – Plotly charts for insights
- 🤖 **AI-Powered Insights** – Gemini-based intelligent explanations

### Advanced Features
- 🧠 **Conversational Memory** – Context-aware multi-turn conversations
- 🌐 **Multi-language Support** – Automatic translation for multilingual users
- 📚 **Business Definitions** – Built-in glossary for key e-commerce terms
- 📦 **Product Enrichment** – Integration of external product knowledge
- 📥 **Data Export** – Download analyzed results as CSV
- 🎨 **Theme Toggle** – Light, Dark, and Pastel UI modes for enhanced UX

---

## 📚 Dataset

The system uses the [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce),  
which contains multiple interconnected tables covering:
- 🛍️ Orders, Customers, and Payments  
- 📦 Products and Categories  
- ⭐ Reviews and Delivery Performance  

This dataset allows the model to perform structured analysis like:
- Identifying top-performing categories  
- Calculating average order values  
- Tracking customer behavior and delivery times  
- Exploring revenue trends and review sentiments  

---

## 🧱 System Architecture

**Workflow:**
1. **User Input** → The user types a natural-language query (e.g., “Top categories by revenue in last 2 quarters”)
2. **Query Understanding** → The Gemini API interprets the question and generates an appropriate SQL query.
3. **Data Retrieval** → SQL is executed on the structured Olist e-commerce dataset (SQLite backend).
4. **Analysis & Visualization** → Results are processed using Pandas and visualized using Plotly.
5. **Response Generation** → Gemini reformulates insights in natural language for clarity.
6. **Conversational Memory** → Streamlit’s session state retains context for multi-turn dialogue.

*(Optionally include a diagram named `architecture.png` to visualize this flow.)*

---

## 🧠 Model & Intelligence

- **Model Used**: Google Gemini 1.5 Flash (via Generative AI API)  
- **Intent Understanding**: Natural language is parsed for relevant fields, time frames, and metrics.  
- **SQL Generation**: Gemini converts parsed intent into optimized SQL queries.  
- **Response Generation**: AI summarizes the result in a conversational tone.  
- **Conversational Memory**: Session memory preserves previous user interactions.  
- **Translation Layer**: Enables multilingual conversations dynamically.  
- **Knowledge Augmentation**: External lookups can enrich responses with product context.  

---

## 🛠️ Tech Stack

| Component | Technology Used |
|------------|-----------------|
| **Framework** | Streamlit |
| **AI Model** | Google Gemini 1.5 Flash |
| **Database** | SQLite (in-memory) |
| **Visualization** | Plotly |
| **Data Processing** | Pandas, NumPy |
| **Environment Management** | Python 3.9+ |
| **Version Control** | Git & GitHub |

---

## 📦 Installation

### Prerequisites
- Python 3.9 or higher  
- [Gemini API Key](https://makersuite.google.com/app/apikey)

### Setup Steps

```bash
# 1. Clone the repository
git clone https://github.com/Sinchana-AS/olistbr_brazilian-ecommerce.git
cd olistbr_brazilian-ecommerce

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 5. Run the Streamlit app
streamlit run 1.py
