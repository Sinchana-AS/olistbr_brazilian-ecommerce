import os
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from datetime import datetime, timedelta
import json
import re
from dotenv import load_dotenv
import requests
from typing import Dict, List, Any

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY not found in .env file!")
    st.info("Please create a .env file with: GEMINI_API_KEY=your_key")
    st.stop()

os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

# Initialize model
try:
    model = genai.GenerativeModel("gemini-1.5-flash")
    print("✅ Gemini model initialized successfully")
except Exception as e:
    try:
        model = genai.GenerativeModel("gemini-pro")
        print("✅ Gemini-pro model initialized")
    except Exception as e2:
        st.error(f"❌ Model initialization error: {e2}")
        model = None


st.set_page_config(page_title="Maersk GenAI Agent", page_icon="🤖", layout="wide")

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

top_cols = st.columns([8, 1])
with top_cols[0]:
    st.markdown('<div class="main-title"></div>', unsafe_allow_html=True)
with top_cols[1]:
    theme_toggle = st.checkbox("🌙", value=(st.session_state.theme=="dark"), key="theme_toggle_small")
st.session_state.theme = "dark" if theme_toggle else "pastel"

def apply_theme(theme):
    if theme == "pastel":
        bg_color = "#fdf6f0"
        chat_bg = "#e8f5e9"
        user_msg_bg = "#a8dadc"
        bot_msg_bg = "#f1faee"
        scrollbar_thumb = "#52b788"
        btn_bg = "#52b788"
        text_color = "#2e3d32"
    else:  # dark
        bg_color = "#1b1b1b"
        chat_bg = "#2c2c2c"
        user_msg_bg = "#52b788"
        bot_msg_bg = "#262626"
        scrollbar_thumb = "#6c63ff"
        btn_bg = "#52b788"
        text_color = "#f5f5f5"
    st.markdown(f"""
    <style>
    html, body, [class*="stAppViewContainer"], [class*="stMain"] {{
        background-color: {bg_color} !important;
        color: {text_color} !important;
        font-family: "Segoe UI", sans-serif;
        overflow: hidden !important;
        height: 100vh !important;
    }}
    
    /* Force sidebar to dark mode */
    section[data-testid="stSidebar"] {{
        background-color: #1b1b1b !important;
    }}
    
    section[data-testid="stSidebar"] * {{
        color: #f5f5f5 !important;
    }}
    
    section[data-testid="stSidebar"] .stButton>button {{
        background-color: #52b788 !important;
        color: white !important;
    }}
    
    section[data-testid="stSidebar"] input {{
        background-color: #2c2c2c !important;
        color: #f5f5f5 !important;
        border: 1px solid #52b788 !important;
    }}
    
    /* Hide scrollbar on main content */
    .main .block-container {{
        overflow: hidden !important;
        max-height: 100vh !important;
        padding-bottom: 2rem !important;
    }}
    
    .main-title {{
        color: {text_color};
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 20px;
        text-align: center;
    }}
    
    .chat-container {{
        height: 500px;
        overflow-y: auto;
        padding: 25px;
        background-color: {chat_bg};
        border: 1px solid #c8e6c9;
        border-radius: 16px;
        margin-bottom: 15px;
        scroll-behavior: smooth;
    }}
    
    /* Chart link styling */
    .chart-link {{
        background: linear-gradient(135deg, #52b788, #429669);
        color: white !important;
        padding: 8px 16px;
        border-radius: 20px;
        text-decoration: none !important;
        font-weight: 600;
        font-size: 14px;
        display: inline-block;
        margin: 5px 0;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(82, 183, 136, 0.3);
        border: none;
        cursor: pointer;
    }}
    
    .chart-link:hover {{
        background: linear-gradient(135deg, #429669, #52b788);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(82, 183, 136, 0.5);
        color: white !important;
        text-decoration: none !important;
    }}
    
    /* History panel styling */
    .history-panel {{
        background-color: {chat_bg};
        border: 1px solid #c8e6c9;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        max-height: 300px;
        overflow-y: auto;
    }}
    
    .history-item {{
        background-color: {bot_msg_bg};
        border: 1px solid #c8e6c9;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
        cursor: pointer;
        transition: all 0.2s ease;
    }}
    
    .history-item:hover {{
        background-color: {user_msg_bg};
        transform: translateX(5px);
    }}
    
    .history-query {{
        font-weight: 600;
        color: {text_color};
        margin-bottom: 5px;
    }}
    
    .history-preview {{
        font-size: 0.85em;
        color: {text_color};
        opacity: 0.8;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    
    /* Make plotly charts fit inline */
    .js-plotly-plot {{
        margin: 10px 0 !important;
        border-radius: 10px !important;
        border: 1px solid #c8e6c9 !important;
    }}
    
    /* Ensure charts are properly styled */
    .stPlotlyChart {{
        margin: 10px 0 !important;
        border-radius: 10px !important;
        border: 1px solid #c8e6c9 !important;
        background-color: {bot_msg_bg} !important;
    }}
    
    /* Style links to look integrated */
    .bot-msg-content a {{
        color: #52b788 !important;
        text-decoration: none !important;
        font-weight: bold !important;
        border-bottom: 1px dotted #52b788 !important;
    }}
    
    .bot-msg-content a:hover {{
        color: #429669 !important;
        border-bottom: 1px solid #429669 !important;
    }}
    
    /* Style expanders to look integrated */
    div[data-testid="stExpander"] {{
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        margin: 10px 0 !important;
    }}
    
    .streamlit-expanderHeader {{
        background-color: rgba(82, 183, 136, 0.2) !important;
        border-radius: 8px !important;
        padding: 10px !important;
        color: {text_color} !important;
    }}
    
    /* Ensure input area stays at bottom */
    .stTextArea {{
        position: relative !important;
    }}
    .user-msg {{ text-align: right; margin: 14px 0; }}
    .user-msg-content {{
        background: {user_msg_bg};
        color: {text_color};
        padding: 18px 22px;
        border-radius: 18px 18px 0 18px;
        display: inline-block;
        max-width: 80%;
        font-size: 17px;
        font-weight: 500;
        box-shadow: 0 0 8px rgba(82, 183, 136, 0.5);
        word-wrap: break-word;
    }}

    .bot-msg {{ text-align: left; margin: 14px 0; }}
    .bot-msg-content {{
        background-color: {bot_msg_bg};
        color: {text_color};
        padding: 18px 22px;
        border-radius: 18px 18px 18px 0;
        display: inline-block;
        max-width: 95%;
        font-size: 17px;
        font-weight: 500;
        box-shadow: 0 0 8px rgba(0,0,0,0.1);
        word-wrap: break-word;
        border: 1px solid #c8e6c9;
    }}

    .thinking-indicator {{
        color: {text_color};
        font-style: italic;
        opacity: 0.7;
        margin: 10px 0;
    }}

    .stTextArea textarea {{
        background-color: {'#2c2c2c' if theme=='dark' else '#ffffff'} !important;
        color: {'#ffffff' if theme=='dark' else text_color} !important;
        border: 1px solid #c8e6c9 !important;
        border-radius: 12px !important;
        font-size: 16px !important;
        padding: 16px 18px !important;
        font-weight: 500 !important;
    }}

    .stButton>button {{
        background: {btn_bg} !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        font-weight: 700 !important;
        padding: 14px 28px !important;
        font-size: 16px !important;
        box-shadow: 0 0 12px rgba(82, 183, 136, 0.5);
        transition: all 0.2s ease-in-out;
        width: 100%;
    }}
    .stButton>button:hover {{ transform: scale(1.05); }}

    ::-webkit-scrollbar {{ width: 10px; }}
    ::-webkit-scrollbar-track {{ background: {chat_bg}; }}
    ::-webkit-scrollbar-thumb {{ background: {scrollbar_thumb}; border-radius: 10px; }}
    </style>
    """, unsafe_allow_html=True)

apply_theme(st.session_state.theme)

# ================================================================
# 🗃 Database Setup
# ================================================================

@st.cache_resource
def setup_lightweight_db():
    """Create a lightweight SQLite database"""
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    
    # Correct way to handle path
    DATA_PATH = os.getenv("DATA_PATH", "C:/Users/ADMIN/Desktop/python/AP/archive")
    
    try:
        print(f"📂 Loading data from: {DATA_PATH}")
        
        orders = pd.read_csv(f"{DATA_PATH}/olist_orders_dataset.csv")
        order_items = pd.read_csv(f"{DATA_PATH}/olist_order_items_dataset.csv")
        products = pd.read_csv(f"{DATA_PATH}/olist_products_dataset.csv")
        customers = pd.read_csv(f"{DATA_PATH}/olist_customers_dataset.csv")
        payments = pd.read_csv(f"{DATA_PATH}/olist_order_payments_dataset.csv")
        trans = pd.read_csv(f"{DATA_PATH}/product_category_name_translation.csv")
        
        print("✅ CSV files loaded successfully")
        
        main_df = (
            orders.merge(order_items, on="order_id", how="inner")
            .merge(products, on="product_id", how="left")
            .merge(customers, on="customer_id", how="left")
            .merge(payments, on="order_id", how="left")
            .merge(trans, on="product_category_name", how="left")
        )
        
        main_df = main_df.sample(min(10000, len(main_df)), random_state=42)
        main_df.to_sql('ecommerce_data', conn, index=False, if_exists='replace')
        
        print(f"✅ Database created: {len(main_df)} records")
        return conn
        
    except Exception as e:
        print(f"⚠️ Error loading CSV files: {e}")
        print("📊 Creating demo data instead...")
        
        # Fixed: Ensure all arrays have same length
        num_records = 1000
        categories = ['bed_bath_table', 'health_beauty', 'sports_leisure', 
                     'computers_accessories', 'furniture_decor', 'housewares',
                     'telephony', 'office_furniture', 'cool_stuff']
        cities = ['sao paulo', 'rio de janeiro', 'belo horizonte', 'brasilia', 'curitiba']
        
        demo_data = pd.DataFrame({
            'product_category_name_english': [categories[i % len(categories)] for i in range(num_records)],
            'price': np.random.uniform(10, 500, num_records),
            'order_purchase_timestamp': pd.date_range('2023-01-01', periods=num_records, freq='8H'),
            'customer_city': [cities[i % len(cities)] for i in range(num_records)],
            'payment_value': np.random.uniform(20, 600, num_records),
            'freight_value': np.random.uniform(5, 50, num_records),
        })
        demo_data.to_sql('ecommerce_data', conn, index=False, if_exists='replace')
        print("✅ Demo database created: 1000 records")
        return conn

db_conn = setup_lightweight_db()

# ================================================================
# 🔍 Database Inspector
# ================================================================

@st.cache_data
def get_database_info():
    """Get database schema"""
    try:
        columns_info = pd.read_sql_query("PRAGMA table_info(ecommerce_data)", db_conn)
        actual_columns = columns_info['name'].tolist()
        
        sample_data = pd.read_sql_query("SELECT * FROM ecommerce_data LIMIT 3", db_conn)
        
        schema_description = "Table: ecommerce_data\nColumns:\n"
        for col in actual_columns:
            sample_val = sample_data[col].iloc[0] if len(sample_data) > 0 else "N/A"
            schema_description += f"  - {col}: {type(sample_val).__name__} (example: {sample_val})\n"
        
        print(f"✅ Database info extracted: {len(actual_columns)} columns")
        return actual_columns, sample_data, schema_description
    except Exception as e:
        print(f"❌ Database info error: {e}")
        return [], pd.DataFrame(), ""

actual_columns, sample_data, schema_description = get_database_info()

# ================================================================
# 🧠 External Knowledge Integration
# ================================================================

class ExternalKnowledge:
    """Simulated external knowledge base for product enrichment"""
    
    @staticmethod
    def get_product_info(category: str) -> Dict[str, Any]:
        """Get enriched product information"""
        knowledge_base = {
            "bed_bath_table": {
                "description": "Bed, bath and table products including linens, towels, and dining accessories",
                "trending": ["Egyptian cotton sheets", "bamboo towels", "ceramic dinnerware"],
                "avg_market_price": "$45-$150",
                "seasonality": "High demand in Q1 (New Year) and Q4 (holidays)"
            },
            "health_beauty": {
                "description": "Health and beauty products including skincare, cosmetics, and wellness items",
                "trending": ["organic skincare", "Korean beauty products", "wellness supplements"],
                "avg_market_price": "$25-$80",
                "seasonality": "Steady year-round, peaks before summer"
            },
            "sports_leisure": {
                "description": "Sports equipment and leisure products for active lifestyles",
                "trending": ["yoga equipment", "home fitness", "outdoor gear"],
                "avg_market_price": "$30-$200",
                "seasonality": "High demand in Q1 (New Year resolutions) and Q2 (summer prep)"
            },
            "computers_accessories": {
                "description": "Computer hardware, peripherals, and tech accessories",
                "trending": ["wireless peripherals", "ergonomic accessories", "RGB gaming gear"],
                "avg_market_price": "$20-$300",
                "seasonality": "Peak in Q4 (holiday shopping) and back-to-school season"
            },
            "furniture_decor": {
                "description": "Home furniture and decorative items",
                "trending": ["minimalist design", "smart furniture", "sustainable materials"],
                "avg_market_price": "$100-$500",
                "seasonality": "High in Q2 (spring remodeling) and Q4 (holidays)"
            }
        }
        
        category_clean = category.lower().replace("_", " ").strip()
        for key, value in knowledge_base.items():
            if key.replace("_", " ") in category_clean or category_clean in key.replace("_", " "):
                return value
        
        return {
            "description": f"E-commerce category: {category}",
            "trending": ["Various popular items"],
            "avg_market_price": "$20-$100",
            "seasonality": "Demand varies throughout the year"
        }
    
    @staticmethod
    def get_market_insights() -> str:
        """Get current market insights"""
        return """
🌍 **Current E-commerce Trends (2024-2025)**:
• Mobile commerce growing 25% YoY
• Sustainability becoming key purchase driver
• Same-day delivery expectations increasing
• AI-powered personalization driving 15% higher conversions
• Social commerce integration expanding rapidly
        """
    
    @staticmethod
    def translate_advanced(text: str, target_lang: str = "portuguese") -> str:
        """Advanced translation with context"""
        if not model:
            return text
        try:
            prompt = f"""Translate the following to {target_lang}. 
Keep e-commerce terminology intact. Only return the translation:

{text}"""
            response = model.generate_content(prompt)
            return response.text.strip()
        except:
            return text

# ================================================================
# 🛠 Agent Tools
# ================================================================

def execute_sql_query(query: str) -> pd.DataFrame:
    """Execute SQL query with error handling"""
    try:
        result = pd.read_sql_query(query, db_conn)
        print(f"✅ Query executed: {len(result)} rows")
        return result
    except Exception as e:
        print(f"❌ Query error: {e}")
        return pd.DataFrame({"error": [str(e)]})

def create_visualization(df: pd.DataFrame, chart_type: str = "bar", title: str = "") -> Any:
    """Create advanced visualizations"""
    try:
        if "error" in df.columns or len(df) == 0:
            return None
            
        if len(df.columns) < 2:
            return None
        
        x_col, y_col = df.columns[0], df.columns[1]
        
        if chart_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, title=title or f"{y_col} by {x_col}",
                        color=y_col, color_continuous_scale="Viridis")
        elif chart_type == "line":
            fig = px.line(df, x=x_col, y=y_col, title=title or f"{y_col} Trend",
                         markers=True)
        elif chart_type == "pie":
            fig = px.pie(df, names=x_col, values=y_col, title=title or "Distribution")
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, title=title or f"{y_col} vs {x_col}",
                           size=y_col if df[y_col].dtype in ['int64', 'float64'] else None)
        else:
            fig = px.bar(df, x=x_col, y=y_col, title=title)
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        print("✅ Visualization created")
        return fig
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        return None

# ================================================================
# 🤖 AI-Powered Agentic System
# ================================================================

class IntelligentAgent:
    """Advanced AI agent with reasoning capabilities"""
    
    def __init__(self, model, schema: str, columns: List[str]):
        self.model = model
        self.schema = schema
        self.columns = columns
        self.external_knowledge = ExternalKnowledge()
    
    def build_context(self, chat_history: List[tuple]) -> str:
        """Build conversation context"""
        if not chat_history:
            return ""
        
        context = "Previous conversation:\n"
        for user_q, bot_resp in chat_history[-3:]:  # Last 3 exchanges
            context += f"User: {user_q}\n"
            context += f"Assistant: {bot_resp[:200]}...\n\n"
        
        return context
    
    def decide_action(self, question: str, context: str = "") -> Dict[str, Any]:
        """AI-powered decision making - handles ANY question intelligently"""
        
        if not self.model:
            print("⚠️ AI model not available, using fallback")
            return self.fallback_decision(question)
        
        prompt = f"""You are an intelligent e-commerce data analyst agent with SQL database access.

**DATABASE SCHEMA:**
{self.schema[:500]}... (showing first 500 chars)

**AVAILABLE COLUMNS:** {', '.join(self.columns[:15])}

**CONVERSATION CONTEXT:**
{context}

**USER QUESTION:** "{question}"

**YOUR TASK:**
Analyze the question and decide what the user wants:

1. **DATA ANALYSIS** - User wants to query/analyze the database
   - Examples: "show me X", "what is the X", "compare X and Y", "find X", "list X"
   - Action: Generate SQL query to answer from database
   
2. **GENERAL ADVICE** - User wants e-commerce advice/knowledge
   - Examples: "how can I X", "what are best practices for X", "tips for X", "improve X"
   - Action: Provide expert consulting advice (NO SQL)
   
3. **PRODUCT INFO** - User wants external knowledge about a product category
   - Examples: "tell me about X category", "info on X products"
   - Action: Fetch external knowledge (NO SQL)

4. **UTILITY** - Translation, definition, explanation
   - Examples: "translate X", "define X", "what does X mean"
   - Action: Use utility tool

**CRITICAL RULES:**
- If question asks about DATA from our database → use SQL_QUERY
- If question asks for ADVICE/KNOWLEDGE → use GENERAL_AI (never SQL)
- Use BOTH if needed (e.g., "show sales AND explain how to improve them")
- Generate VALID SQLite syntax only
- Use actual column names from schema
- LIMIT all queries to 15 rows

**OUTPUT FORMAT (valid JSON only):**
{{
    "reasoning": "brief explanation of your decision",
    "tools": ["SQL_QUERY" | "GENERAL_AI" | "EXTERNAL_KNOWLEDGE" | "TRANSLATION"],
    "sql_query": "SELECT ... FROM ecommerce_data ..." or null,
    "chart_type": "bar" | "line" | "pie" | null,
    "needs_external_knowledge": true | false,
    "category_lookup": "category_name" or null,
    "general_response": true | false,
    "general_question": "reformulated question" or null
}}

**EXAMPLES:**

Question: "Which products sold the most last month?"
Response: {{"reasoning": "Data query about sales", "tools": ["SQL_QUERY", "VISUALIZATION"], "sql_query": "SELECT product_category_name_english, SUM(price) as revenue FROM ecommerce_data GROUP BY product_category_name_english ORDER BY revenue DESC LIMIT 10", "chart_type": "bar", "general_response": false}}

Question: "How can I boost my conversion rate?"
Response: {{"reasoning": "General e-commerce advice", "tools": ["GENERAL_AI"], "sql_query": null, "general_response": true, "general_question": "How to improve e-commerce conversion rate?"}}

Question: "Show revenue trends then explain how to improve them"
Response: {{"reasoning": "Needs both data and advice", "tools": ["SQL_QUERY", "VISUALIZATION", "GENERAL_AI"], "sql_query": "SELECT strftime('%Y-%m', order_purchase_timestamp) as month, SUM(price) as revenue FROM ecommerce_data GROUP BY month ORDER BY month LIMIT 12", "chart_type": "line", "general_response": true, "general_question": "How to improve revenue trends?"}}

Now analyze: "{question}"
"""
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
                print(f"✅ AI Decision: {decision.get('reasoning', 'No reasoning')}")
                print(f"🔧 Tools: {decision.get('tools', [])}")
                return decision
            else:
                print("⚠️ Could not parse AI response, using smart fallback")
                return self.fallback_decision(question)
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON parse error: {e}")
            return self.fallback_decision(question)
        except Exception as e:
            print(f"❌ AI decision error: {e}")
            return self.fallback_decision(question)
    
    def fallback_decision(self, question: str) -> Dict[str, Any]:
        """Smart fallback when AI fails - uses semantic understanding"""
        q = question.lower()
        
        # Check if it's a GENERAL question (advice/knowledge - NOT data)
        general_indicators = [
            'how to', 'how can', 'how do', 'how should', 'how would',
            'what is', 'what are', 'what\'s', 'why', 'explain',
            'best practice', 'best way', 'improve', 'increase', 'boost', 'enhance',
            'optimize', 'strategy', 'strategies', 'advice', 'recommend', 'suggest',
            'tips', 'guide', 'help', 'reduce', 'decrease', 'prevent', 'avoid',
            'meaning', 'difference between', 'better', 'should i'
        ]
        
        # Check if it's a DATA query (asking for information FROM database)
        data_indicators = [
            'show', 'display', 'list', 'get', 'fetch', 'find', 'search',
            'what', 'which', 'how many', 'how much', 'count', 'sum', 'average',
            'top', 'bottom', 'highest', 'lowest', 'most', 'least', 'best selling',
            'compare', 'versus', 'vs', 'between', 'trend', 'over time',
            'by category', 'by city', 'by month', 'revenue', 'sales', 'orders'
        ]
        
        # Score the question
        general_score = sum(1 for indicator in general_indicators if indicator in q)
        data_score = sum(1 for indicator in data_indicators if indicator in q)
        
        print(f"🎯 Fallback scores - General: {general_score}, Data: {data_score}")
        
        # Decide based on scores
        if general_score > data_score:
            # It's a GENERAL question
            return {
                "reasoning": "General e-commerce question requiring expert advice (fallback)",
                "tools": ["GENERAL_AI"],
                "sql_query": None,
                "chart_type": None,
                "needs_external_knowledge": False,
                "general_response": True,
                "general_question": question
            }
        
        # DATA QUERIES - Try to generate intelligent SQL
        
        # Schema/Structure query
        if any(word in q for word in ['column', 'field', 'schema', 'table', 'structure', 'available']):
            return {
                "reasoning": "User wants database structure info",
                "tools": ["DEFINITION"],
                "sql_query": None,
                "chart_type": None,
                "needs_external_knowledge": False,
                "general_response": False
            }
        
        # Aggregation queries (top, highest, most, best)
        if any(word in q for word in ['top', 'best', 'highest', 'most', 'greatest']):
            # Check what they want top of
            if 'categor' in q:
                sql = """SELECT product_category_name_english, SUM(price) as total_revenue 
                         FROM ecommerce_data WHERE product_category_name_english IS NOT NULL 
                         GROUP BY product_category_name_english ORDER BY total_revenue DESC LIMIT 10"""
                return {
                    "reasoning": "Top categories query (fallback)",
                    "tools": ["SQL_QUERY", "VISUALIZATION"],
                    "sql_query": sql,
                    "chart_type": "bar",
                    "needs_external_knowledge": False,
                    "general_response": False
                }
            elif any(word in q for word in ['city', 'cities', 'location']):
                sql = """SELECT customer_city, COUNT(*) as orders, SUM(price) as revenue 
                         FROM ecommerce_data WHERE customer_city IS NOT NULL 
                         GROUP BY customer_city ORDER BY revenue DESC LIMIT 10"""
                return {
                    "reasoning": "Top cities query (fallback)",
                    "tools": ["SQL_QUERY", "VISUALIZATION"],
                    "sql_query": sql,
                    "chart_type": "bar",
                    "needs_external_knowledge": False,
                    "general_response": False
                }
            else:
                # Generic top query
                sql = """SELECT product_category_name_english, SUM(price) as total 
                         FROM ecommerce_data WHERE product_category_name_english IS NOT NULL 
                         GROUP BY product_category_name_english ORDER BY total DESC LIMIT 10"""
                return {
                    "reasoning": "Generic top query (fallback)",
                    "tools": ["SQL_QUERY", "VISUALIZATION"],
                    "sql_query": sql,
                    "chart_type": "bar",
                    "needs_external_knowledge": False,
                    "general_response": False
                }
        
        # Trend/Time-based queries
        if any(word in q for word in ['trend', 'over time', 'monthly', 'quarterly', 'time', 'period']):
            sql = """SELECT strftime('%Y-%m', order_purchase_timestamp) as month, 
                     SUM(price) as revenue FROM ecommerce_data 
                     WHERE order_purchase_timestamp IS NOT NULL 
                     GROUP BY month ORDER BY month DESC LIMIT 12"""
            return {
                "reasoning": "Time trend query (fallback)",
                "tools": ["SQL_QUERY", "VISUALIZATION"],
                "sql_query": sql,
                "chart_type": "line",
                "needs_external_knowledge": False,
                "general_response": False
            }
        
        # Average queries
        if any(word in q for word in ['average', 'avg', 'mean']):
            sql = """SELECT product_category_name_english, AVG(price) as avg_price 
                     FROM ecommerce_data WHERE product_category_name_english IS NOT NULL 
                     GROUP BY product_category_name_english ORDER BY avg_price DESC LIMIT 10"""
            return {
                "reasoning": "Average calculation query (fallback)",
                "tools": ["SQL_QUERY", "VISUALIZATION"],
                "sql_query": sql,
                "chart_type": "bar",
                "needs_external_knowledge": False,
                "general_response": False
            }
        
        # Comparison queries
        if any(word in q for word in ['compare', 'comparison', 'versus', 'vs', 'between']):
            # Try to extract city names
            cities = ['sao paulo', 'rio', 'brasilia', 'belo horizonte', 'curitiba', 'paulo', 'janeiro']
            found_cities = [city for city in cities if city in q]
            
            if found_cities:
                sql = """SELECT customer_city, COUNT(*) as orders, SUM(price) as revenue 
                         FROM ecommerce_data WHERE customer_city IS NOT NULL 
                         GROUP BY customer_city ORDER BY revenue DESC LIMIT 10"""
            else:
                sql = """SELECT product_category_name_english, SUM(price) as revenue 
                         FROM ecommerce_data WHERE product_category_name_english IS NOT NULL 
                         GROUP BY product_category_name_english ORDER BY revenue DESC LIMIT 10"""
            
            return {
                "reasoning": "Comparison query (fallback)",
                "tools": ["SQL_QUERY", "VISUALIZATION"],
                "sql_query": sql,
                "chart_type": "bar",
                "needs_external_knowledge": False,
                "general_response": False
            }
        
        # Location/City queries
        if any(word in q for word in ['city', 'cities', 'location', 'where', 'region']):
            sql = """SELECT customer_city, COUNT(*) as orders, SUM(price) as revenue 
                     FROM ecommerce_data WHERE customer_city IS NOT NULL 
                     GROUP BY customer_city ORDER BY revenue DESC LIMIT 10"""
            return {
                "reasoning": "Location-based query (fallback)",
                "tools": ["SQL_QUERY", "VISUALIZATION"],
                "sql_query": sql,
                "chart_type": "bar",
                "needs_external_knowledge": False,
                "general_response": False
            }
        
        # Product category info
        if any(word in q for word in ['about', 'tell me', 'info', 'information', 'detail']):
            categories = ['health_beauty', 'bed_bath_table', 'sports_leisure', 
                         'computers_accessories', 'furniture_decor', 'housewares']
            for cat in categories:
                if cat.replace('_', ' ') in q or cat in q:
                    return {
                        "reasoning": f"External knowledge about {cat} (fallback)",
                        "tools": ["EXTERNAL_KNOWLEDGE"],
                        "sql_query": None,
                        "chart_type": None,
                        "needs_external_knowledge": True,
                        "category_lookup": cat,
                        "general_response": False
                    }
        
        # Count queries
        if any(word in q for word in ['how many', 'count', 'number of', 'total']):
            if 'order' in q:
                sql = """SELECT COUNT(DISTINCT order_id) as total_orders FROM ecommerce_data"""
            elif 'categor' in q:
                sql = """SELECT COUNT(DISTINCT product_category_name_english) as categories FROM ecommerce_data"""
            elif 'custom' in q:
                sql = """SELECT COUNT(DISTINCT customer_id) as customers FROM ecommerce_data"""
            else:
                sql = """SELECT COUNT(*) as total_records FROM ecommerce_data"""
            
            return {
                "reasoning": "Count query (fallback)",
                "tools": ["SQL_QUERY"],
                "sql_query": sql,
                "chart_type": None,
                "needs_external_knowledge": False,
                "general_response": False
            }
        
        # Last resort: If nothing matches but looks like data query, show sample
        if data_score > 0:
            sql = f"SELECT {', '.join(self.columns[:5])} FROM ecommerce_data LIMIT 10"
            return {
                "reasoning": "Showing sample data (fallback - couldn't determine specific query)",
                "tools": ["SQL_QUERY"],
                "sql_query": sql,
                "chart_type": None,
                "needs_external_knowledge": False,
                "general_response": False
            }
        
        # Absolute last resort: General AI response
        return {
            "reasoning": "Unclear question - providing general response (fallback)",
            "tools": ["GENERAL_AI"],
            "sql_query": None,
            "chart_type": None,
            "needs_external_knowledge": False,
            "general_response": True,
            "general_question": question
        }
    
    def execute_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's decision"""
        result = {
            "response": f"🤔 **Reasoning:** {decision.get('reasoning', 'Processing your request...')}\n\n",
            "data": None,
            "visualization": None,
            "external_info": None,
            "chart_type": decision.get("chart_type"),
            "query": decision.get("sql_query")
        }
        
        # Handle DEFINITION tool (show columns)
        if "DEFINITION" in decision.get("tools", []):
            result["response"] += f"**📋 Available Columns** ({len(self.columns)}):\n\n"
            result["response"] += "\n".join([f"{i+1}. `{col}`" for i, col in enumerate(self.columns[:20])])
            if len(self.columns) > 20:
                result["response"] += f"\n\n_...and {len(self.columns)-20} more columns_"
            return result
        
        # Handle General AI Response (for advice, explanations, etc.)
        if decision.get("general_response") and decision.get("general_question"):
            if self.model:
                try:
                    general_prompt = f"""You are an experienced e-commerce consultant with expertise in:
- Conversion rate optimization
- Customer experience and UX design  
- Digital marketing strategies
- Data analytics and insights
- Supply chain and logistics
- Customer retention and loyalty

Question: {decision['general_question']}

Provide a comprehensive, actionable answer with:
1. Clear explanation
2. Specific strategies or recommendations  
3. Real-world examples where relevant
4. 2-3 key takeaways

Keep your response well-structured and under 250 words."""
                    
                    response = self.model.generate_content(general_prompt)
                    result["response"] += f"💡 **E-commerce Expert Answer:**\n\n{response.text.strip()}\n"
                    
                    # Add helpful note
                    result["response"] += f"\n\n_💬 Want data to back this up? Try asking: \"Show me relevant data from our database\"_"
                    
                except Exception as e:
                    print(f"General AI error: {e}")
                    result["response"] += f"\n⚠️ I can help with that, but I'm having trouble generating a response right now. Could you rephrase your question?\n"
            else:
                result["response"] += "\n⚠️ AI model not available. Please check your API key.\n"
            return result
        
        # Execute SQL Query
        if "SQL_QUERY" in decision.get("tools", []) and decision.get("sql_query"):
            df = execute_sql_query(decision["sql_query"])
            
            if "error" not in df.columns and len(df) > 0:
                result["data"] = df
                result["response"] += "📊 **Query Results:**\n\n"
                
                # Format results nicely
                for idx, row in df.head(10).iterrows():
                    line = " | ".join([
                        f"**{k}**: {v:,.2f}" if isinstance(v, (int, float)) and not pd.isna(v)
                        else f"**{k}**: {v}" 
                        for k, v in row.items() if pd.notna(v)
                    ])
                    result["response"] += f"• {line}\n"
                
                if len(df) > 10:
                    result["response"] += f"\n_...and {len(df)-10} more results_\n"
            else:
                result["response"] += f"\n⚠️ Query returned no results or had an error.\n"
        
        # Create Visualization - ALWAYS create if visualization is in tools
        if "VISUALIZATION" in decision.get("tools", []) and result["data"] is not None:
            chart_type = decision.get("chart_type", "bar")
            viz = create_visualization(result["data"], chart_type)
            if viz:
                result["visualization"] = viz
                # Add clickable chart link that routes to chart gallery
                chart_index = len([d for d in st.session_state.chat_data if d.get("visualization")])
                result["response"] += f'\n\n<button class="chart-link" onclick="window.streamlitBridge.setComponentValue({{chart_index: {chart_index}}})">📊 View Interactive Chart</button>'
        
        # External Knowledge
        if decision.get("needs_external_knowledge") and decision.get("category_lookup"):
            category = decision["category_lookup"]
            info = self.external_knowledge.get_product_info(category)
            result["external_info"] = info
            result["response"] += f"\n\n🌐 **External Knowledge - {category.replace('_', ' ').title()}:**\n"
            result["response"] += f"• **Description:** {info['description']}\n"
            result["response"] += f"• **Trending Items:** {', '.join(info['trending'])}\n"
            result["response"] += f"• **Market Price Range:** {info['avg_market_price']}\n"
            result["response"] += f"• **Seasonality:** {info['seasonality']}\n"
        
        # Translation
        if decision.get("needs_translation") and decision.get("text_to_translate"):
            translated = self.external_knowledge.translate_advanced(decision["text_to_translate"])
            result["response"] += f"\n\n🌐 **Translation (Portuguese):**\n{translated}\n"
        
        return result

# ================================================================
# 💬 Session State
# ================================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_data" not in st.session_state:
    st.session_state.chat_data = []
if "user_preferences" not in st.session_state:
    st.session_state.user_preferences = {"favorite_categories": [], "preferred_cities": []}
if "agent" not in st.session_state:
    st.session_state.agent = IntelligentAgent(model, schema_description, actual_columns)
if "current_page" not in st.session_state:
    st.session_state.current_page = "chat"
if "selected_chart_index" not in st.session_state:
    st.session_state.selected_chart_index = None
if "chart_clicked" not in st.session_state:
    st.session_state.chart_clicked = None

# ================================================================
# 📊 Chart Gallery Page
# ================================================================

def show_chart_gallery():
    """Display the chart gallery page"""
    st.markdown(f'<div class="main-title">📊 Chart Gallery</div>', unsafe_allow_html=True)
    
    # Get all conversations with charts
    chart_conversations = []
    for i, (query, data) in enumerate(zip(st.session_state.chat_history, st.session_state.chat_data)):
        if data.get("visualization"):
            chart_conversations.append({
                "index": i,
                "query": query[0] if isinstance(query, tuple) else query,  # Handle both tuple and string formats
                "chart_type": data.get("chart_type", "bar"),
                "data": data.get("data"),
                "visualization": data.get("visualization")
            })
    
    if not chart_conversations:
        st.info("📈 No charts generated yet. Ask questions that create visualizations!")
        st.markdown("**Try asking:**")
        st.markdown("- 'Show top categories with chart'")
        st.markdown("- 'Display revenue trends over time'") 
        st.markdown("- 'Compare sales by city with visualization'")
        
        if st.button("🔙 Back to Chat", use_container_width=True):
            st.session_state.current_page = "chat"
            st.rerun()
        return
    
    # Display selected chart or gallery
    if st.session_state.selected_chart_index is not None:
        show_single_chart(st.session_state.selected_chart_index)
    else:
        show_chart_gallery_grid(chart_conversations)

def show_single_chart(chart_index):
    """Display a single chart in full view"""
    # Find the actual data index in chat_data
    actual_index = None
    chart_count = 0
    
    for i, data in enumerate(st.session_state.chat_data):
        if data.get("visualization"):
            if chart_count == chart_index:
                actual_index = i
                break
            chart_count += 1
    
    if actual_index is not None and actual_index < len(st.session_state.chat_data):
        data = st.session_state.chat_data[actual_index]
        query = st.session_state.chat_history[actual_index]
        query_text = query[0] if isinstance(query, tuple) else query
        
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"### 📈 {query_text}")
        with col2:
            if st.button("🔙 Back to Gallery", use_container_width=True):
                st.session_state.selected_chart_index = None
                st.rerun()
        
        # Display the chart
        if data.get("visualization"):
            # Update chart layout for better display
            full_fig = data["visualization"]
            full_fig.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=60, b=20),
                template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                title_font_size=20
            )
            st.plotly_chart(full_fig, use_container_width=True)
        
        # Show data table
        if data.get("data") is not None:
            with st.expander("📋 View Data Table", expanded=False):
                st.dataframe(data["data"], use_container_width=True)
        
        # Chart details
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Chart Type", data.get("chart_type", "bar").title())
        with col2:
            st.metric("Data Points", len(data["data"]) if data.get("data") is not None else 0)
        with col3:
            st.metric("Generated", "Just now")
    
    else:
        st.error("❌ Chart not found")
        if st.button("🔙 Back to Gallery", use_container_width=True):
            st.session_state.selected_chart_index = None
            st.rerun()

def show_chart_gallery_grid(chart_conversations):
    """Display chart gallery as a grid"""
    st.markdown(f"### 🖼️ Your Generated Charts ({len(chart_conversations)} total)")
    
    # Search and filter
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("🔍 Search charts...", placeholder="Search by query or chart type")
    with col2:
        chart_types = ["All"] + list(set([conv["chart_type"] for conv in chart_conversations]))
        selected_type = st.selectbox("Filter by type", chart_types)
    
    # Filter charts
    filtered_charts = [
        conv for conv in chart_conversations
        if (not search_query or search_query.lower() in conv["query"].lower()) and
           (selected_type == "All" or conv["chart_type"] == selected_type)
    ]
    
    if not filtered_charts:
        st.info("🔍 No charts match your search criteria.")
        return
    
    # Display as grid
    cols = st.columns(2)
    for i, chart in enumerate(filtered_charts):
        with cols[i % 2]:
            with st.container():
                st.markdown(f"**📊 {chart['query'][:80]}{'...' if len(chart['query']) > 80 else ''}**")
                
                # Display mini chart preview
                if chart["visualization"]:
                    # Create a smaller version of the chart for preview
                    mini_fig = chart["visualization"]
                    mini_fig.update_layout(
                        height=200, 
                        margin=dict(l=20, r=20, t=30, b=20),
                        showlegend=False,
                        title=""
                    )
                    st.plotly_chart(mini_fig, use_container_width=True, config={'displayModeBar': False})
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔍 View Full", key=f"view_{chart['index']}", use_container_width=True):
                        st.session_state.selected_chart_index = i  # Use gallery index, not chat index
                        st.rerun()
                with col2:
                    st.caption(f"📈 {chart['chart_type'].title()}")
    
    st.markdown("---")
    if st.button("🔙 Back to Chat", use_container_width=True):
        st.session_state.current_page = "chat"
        st.rerun()

# ================================================================
# 📄 History Page
# ================================================================

def show_history_page():
    """Display the conversation history page"""
    st.markdown(f'<div class="main-title">📚 Conversation History</div>', unsafe_allow_html=True)
    
    if not st.session_state.chat_history:
        st.info("No conversation history yet. Start chatting to build history!")
        if st.button("🔙 Back to Chat", use_container_width=True):
            st.session_state.current_page = "chat"
            st.rerun()
        return
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Conversations", len(st.session_state.chat_history))
    with col2:
        total_queries = len([data for data in st.session_state.chat_data if "SQL_QUERY" in str(data)])
        st.metric("Data Queries", total_queries)
    with col3:
        total_charts = len([data for data in st.session_state.chat_data if data.get("visualization")])
        st.metric("Charts Generated", total_charts)
    with col4:
        if st.button("📊 Chart Gallery", use_container_width=True):
            st.session_state.current_page = "charts"
            st.rerun()
    
    st.markdown("---")
    
    # Search and filter
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("🔍 Search conversations...", placeholder="Type to search your chat history")
    with col2:
        if st.button("📥 Export All", use_container_width=True):
            export_conversation_data()
    
    # Display conversation history
    for i, (query, data) in enumerate(zip(st.session_state.chat_history, st.session_state.chat_data)):
        # Filter by search query
        if search_query and search_query.lower() not in query.lower():
            continue
            
        with st.expander(f"💬 {query}", expanded=False):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**Query:** {query}")
                st.markdown(f"**Response:** {data.get('response', 'No response')}")
                
                if data.get('visualization'):
                    st.success("📊 Includes chart visualization")
                    if st.button("📈 View Chart", key=f"history_chart_{i}", use_container_width=True):
                        st.session_state.selected_chart_index = i
                        st.session_state.current_page = "charts"
                        st.rerun()
                
                if data.get('data') is not None:
                    df = data['data']
                    st.info(f"📈 Data results: {len(df)} rows")
                    
            with col2:
                if st.button("🔄 Re-run", key=f"rerun_{i}", use_container_width=True):
                    st.session_state.pending_query = query
                    st.session_state.current_page = "chat"
                    st.rerun()
    
    st.markdown("---")
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🔙 Back to Chat", use_container_width=True):
            st.session_state.current_page = "chat"
            st.rerun()
    with col2:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.chat_data = []
            st.rerun()
    with col3:
        if st.button("📈 Analytics", use_container_width=True):
            show_history_analytics()

def export_conversation_data():
    """Export conversation data as JSON"""
    export_data = []
    for i, (query, data) in enumerate(zip(st.session_state.chat_history, st.session_state.chat_data)):
        export_data.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "response": data.get("response", ""),
            "has_visualization": data.get("visualization") is not None,
            "has_data": data.get("data") is not None
        })
    
    export_json = json.dumps(export_data, indent=2)
    st.download_button(
        label="💾 Download JSON",
        data=export_json,
        file_name=f"maersk_chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )

def show_history_analytics():
    """Show analytics about conversation history"""
    st.subheader("📈 Conversation Analytics")
    
    if not st.session_state.chat_history:
        st.info("No data to analyze")
        return
    
    # Basic stats
    total_conversations = len(st.session_state.chat_history)
    avg_query_length = np.mean([len(q) for q in st.session_state.chat_history])
    
    # Query type analysis
    query_types = {
        "Data Queries": len([data for data in st.session_state.chat_data if "SQL_QUERY" in str(data)]),
        "Charts": len([data for data in st.session_state.chat_data if data.get("visualization")]),
        "General Advice": len([data for data in st.session_state.chat_data if "GENERAL_AI" in str(data)]),
        "External Knowledge": len([data for data in st.session_state.chat_data if data.get("external_info")])
    }
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Chats", total_conversations)
    col2.metric("Avg Query Length", f"{avg_query_length:.1f} chars")
    col3.metric("Data Queries", query_types["Data Queries"])
    col4.metric("Charts Generated", query_types["Charts"])
    
    # Query type distribution
    fig = px.pie(
        values=list(query_types.values()),
        names=list(query_types.keys()),
        title="Query Type Distribution",
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig, use_container_width=True)

# ================================================================
# 🎨 Sidebar
# ================================================================

with st.sidebar:
    st.title("🚢 AI-powered agent")
    st.markdown("---")
    
    # Page Navigation - WITH CHARTS BUTTON
    st.subheader("🧭 Navigation")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💬 Chat", use_container_width=True, type="primary" if st.session_state.current_page == "chat" else "secondary"):
            st.session_state.current_page = "chat"
            st.rerun()
    with col2:
        if st.button("📊 Charts", use_container_width=True, type="primary" if st.session_state.current_page == "charts" else "secondary"):
            st.session_state.current_page = "charts"
            st.rerun()
    with col3:
        if st.button("📚 History", use_container_width=True, type="primary" if st.session_state.current_page == "history" else "secondary"):
            st.session_state.current_page = "history"
            st.rerun()
    
    st.markdown("---")
    
    # Debug Mode Toggle (only show in chat page)
    if st.session_state.current_page == "chat":
        debug_mode = st.checkbox("🐛 Debug Mode", value=False, help="Show AI decision making process")
    
    # User Personalization
    st.subheader("👤 Personalization")
    user_name = st.text_input("Your Name (optional):", value=st.session_state.get("user_name", ""))
    if user_name:
        st.session_state.user_name = user_name
    
    st.markdown("---")
    st.subheader("📊 Database Info")
    st.write(f"**Total Columns:** {len(actual_columns)}")
    st.write(f"**Records:** ~10,000")
    
    with st.expander("🔍 Schema Details"):
        for i, col in enumerate(actual_columns[:15], 1):
            st.text(f"{i}. {col}")
        if len(actual_columns) > 15:
            st.text(f"... and {len(actual_columns)-15} more")
    
    # Quick Actions (only show in chat page)
    if st.session_state.current_page == "chat":
        st.markdown("---")
        st.subheader("🚀 Quick Actions")
        
        quick_queries = [
            "Show me all columns",
            "Top 5 selling categories",
            "Revenue trends last 6 months",
            "Compare São Paulo vs Rio sales",
            "Average order value by category",
            "Tell me about health_beauty products",
        ]
        
        for query in quick_queries:
            if st.button(f"💡 {query}", use_container_width=True, key=f"quick_{query}"):
                st.session_state.pending_query = query
                st.rerun()
    
    st.markdown("---")
    
    # Quick History Preview (only show in chat page)
    if st.session_state.current_page == "chat" and st.session_state.chat_history:
        st.subheader("💬 Recent History")
        
        # Show last 5 queries
        recent_queries = list(reversed(st.session_state.chat_history[-5:]))
        
        for i, query in enumerate(recent_queries):
            if st.button(f"🔁 {query[:50]}...", key=f"recent_{i}", use_container_width=True):
                st.session_state.pending_query = query
                st.rerun()
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.chat_data = []
        st.rerun()

# ================================================================
# 📄 Main Chat Interface
# ================================================================

if st.session_state.current_page == "chat":
    greeting = f"👋 Hello{' ' + st.session_state.get('user_name', '') if st.session_state.get('user_name') else ''}! I'm your intelligent E-commerce analyst!"
    st.markdown(f'<div class="main-title">🤖 {greeting}</div>', unsafe_allow_html=True)

    # Chat display container
    chat_container = st.container()

    with chat_container:
        chat_html = '<div class="chat-container">'
        
        if not st.session_state.chat_history:
            chat_html += '''<div class="bot-msg"><div class="bot-msg-content">
            🚀 I'm an AI-powered agent, How can I help you:<br><br
            </div></div>'''
        
        for i, (user_q, bot_data) in enumerate(zip(st.session_state.chat_history, st.session_state.chat_data)):
            # User message
            chat_html += f'<div class="user-msg"><div class="user-msg-content">🧑 {user_q}</div></div>'
            
            # Bot response - with clickable chart links
            bot_response = bot_data.get("response", "Processing...")
            
            # Replace the chart link with a clickable button that routes to chart gallery
            if bot_data.get("visualization"):
                # Find the chart index for this conversation
                chart_index = len([d for d in st.session_state.chat_data[:i] if d.get("visualization")])
                # Create a clickable button that will route to the chart gallery
                chart_button = f'''
                <div style="margin: 10px 0;">
                    <button class="chart-link" onclick="window.parent.postMessage({{'type': 'streamlit:setComponentValue', 'value': {{'chart_index': {chart_index}, 'page': 'charts'}}}}, '*')">
                        📊 View Interactive Chart
                    </button>
                </div>
                '''
                # Replace the placeholder with the actual button
                bot_response = bot_response.replace('[📊 View Interactive Chart](#chart-gallery)', chart_button)
            
            chat_html += f'<div class="bot-msg"><div class="bot-msg-content">{bot_response}</div></div>'
        
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)

    # ================================================================
    # 💬 Input Handler
    # ================================================================

    st.markdown("---")
    col1, col2 = st.columns([5, 1])

    with col1:
        user_input = st.text_area(
            "Ask anything...", 
            height=80, 
            key=f"input_{len(st.session_state.chat_history)}",
            placeholder="e.g., 'Show top categories', then 'Now compare with last quarter'"
        )

    with col2:
        st.write("")
        st.write("")
        send_button = st.button("Send 💬", type="primary", use_container_width=True)

    # Handle pending query from quick actions
    if "pending_query" in st.session_state:
        user_input = st.session_state.pending_query
        send_button = True
        del st.session_state.pending_query

    # Handle chart click events
    if st.session_state.get("chart_clicked") is not None:
        chart_index = st.session_state.chart_clicked
        st.session_state.selected_chart_index = chart_index
        st.session_state.current_page = "charts"
        st.session_state.chart_clicked = None
        st.rerun()

    if send_button and user_input.strip():
        with st.spinner("🤔 Thinking..."):
            # Build context from conversation history
            context = st.session_state.agent.build_context(
                list(zip(st.session_state.chat_history, 
                        [d.get("response", "") for d in st.session_state.chat_data]))
            )
            
            # AI-powered decision making
            try:
                decision = st.session_state.agent.decide_action(user_input, context)
                print(f"🔍 Decision made: {decision.get('tools', [])}")
                print(f"📝 Reasoning: {decision.get('reasoning', 'N/A')}")
                print(f"📊 Chart type: {decision.get('chart_type', 'N/A')}")
                
                # Show debug info if enabled
                if 'debug_mode' in locals() and debug_mode:
                    st.info(f"""
    **🐛 Debug Info:**
    - **Tools Selected:** {', '.join(decision.get('tools', ['None']))}
    - **Reasoning:** {decision.get('reasoning', 'N/A')}
    - **SQL Query:** {decision.get('sql_query', 'None')[:100] if decision.get('sql_query') else 'None'}
    - **Chart Type:** {decision.get('chart_type', 'None')}
    - **General AI:** {decision.get('general_response', False)}
                    """)
                    
            except Exception as e:
                print(f"❌ Decision error: {e}")
                st.error(f"⚠️ Decision making error: {e}")
                decision = st.session_state.agent.fallback_decision(user_input)
            
            # Execute the decision
            try:
                result = st.session_state.agent.execute_decision(decision)
                print(f"✅ Execution result - Has visualization: {result.get('visualization') is not None}")
                print(f"✅ Execution result - Has data: {result.get('data') is not None}")
            except Exception as e:
                print(f"❌ Execution error: {e}")
                st.error(f"⚠️ Execution error: {e}")
                result = {
                    "response": f"⚠️ I encountered an error: {str(e)}\n\nPlease try rephrasing your question or check the debug mode.",
                    "data": None,
                    "visualization": None
                }
            
            # Update chat history
            st.session_state.chat_history.append(user_input)
            st.session_state.chat_data.append(result)
            
            # Update user preferences based on queries
            if decision.get("category_lookup"):
                cat = decision["category_lookup"]
                if cat not in st.session_state.user_preferences["favorite_categories"]:
                    st.session_state.user_preferences["favorite_categories"].append(cat)
            
        st.rerun()

    # ================================================================
    # 📊 Analytics Dashboard (Bonus Feature)
    # ================================================================

    with st.expander("📈 Auto-Generated Insights Dashboard", expanded=False):
        st.subheader("🎯 Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        try:
            # Total Revenue
            revenue_df = execute_sql_query("SELECT SUM(price) as total FROM ecommerce_data")
            total_revenue = revenue_df['total'].iloc[0] if len(revenue_df) > 0 else 0
            col1.metric("💰 Total Revenue", f"${total_revenue:,.0f}")
            
            # Total Orders
            orders_df = execute_sql_query("SELECT COUNT(DISTINCT order_id) as total FROM ecommerce_data WHERE order_id IS NOT NULL")
            total_orders = orders_df['total'].iloc[0] if len(orders_df) > 0 else 0
            col2.metric("📦 Total Orders", f"{total_orders:,}")
            
            # Avg Order Value
            avg_order = total_revenue / total_orders if total_orders > 0 else 0
            col3.metric("💵 Avg Order Value", f"${avg_order:.2f}")
            
            # Categories
            cat_df = execute_sql_query("SELECT COUNT(DISTINCT product_category_name_english) as total FROM ecommerce_data")
            total_cats = cat_df['total'].iloc[0] if len(cat_df) > 0 else 0
            col4.metric("🏷️ Categories", f"{total_cats}")
            
        except Exception as e:
            st.error(f"Could not load metrics: {e}")
        
        # Quick Insights
        st.markdown("---")
        st.subheader("💡 AI-Generated Insights")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.markdown("**🔥 Top Performing:**")
            try:
                top_cat_df = execute_sql_query("""
                    SELECT product_category_name_english, SUM(price) as revenue 
                    FROM ecommerce_data 
                    WHERE product_category_name_english IS NOT NULL
                    GROUP BY product_category_name_english 
                    ORDER BY revenue DESC LIMIT 3
                """)
                for _, row in top_cat_df.iterrows():
                    st.write(f"• {row['product_category_name_english']}: ${row['revenue']:,.0f}")
            except:
                st.write("• Data unavailable")
        
        with insights_col2:
            st.markdown("**📍 Top Cities:**")
            try:
                top_city_df = execute_sql_query("""
                    SELECT customer_city, COUNT(*) as orders 
                    FROM ecommerce_data 
                    WHERE customer_city IS NOT NULL
                    GROUP BY customer_city 
                    ORDER BY orders DESC LIMIT 3
                """)
                for _, row in top_city_df.iterrows():
                    st.write(f"• {row['customer_city']}: {row['orders']:,} orders")
            except:
                st.write("• Data unavailable")
        
        # Market Insights
        st.markdown("---")
        st.markdown(ExternalKnowledge.get_market_insights())

    # ================================================================
    # 🎯 Smart Recommendations
    # ================================================================

    if len(st.session_state.chat_history) > 0:
        with st.expander("🎯 Smart Recommendations for You", expanded=False):
            st.markdown("**Based on your conversation, you might also want to:**")
            
            recommendations = []
            
            # Analyze recent queries
            recent_queries = " ".join(st.session_state.chat_history[-3:]).lower()
            
            if "category" in recent_queries or "top" in recent_queries:
                recommendations.append("💡 Compare category performance across different cities")
                recommendations.append("💡 Analyze seasonal trends for top categories")
            
            if "city" in recent_queries or "location" in recent_queries:
                recommendations.append("💡 See delivery time variations by region")
                recommendations.append("💡 Compare urban vs suburban purchasing patterns")
            
            if "trend" in recent_queries or "time" in recent_queries:
                recommendations.append("💡 Forecast next quarter's revenue")
                recommendations.append("💡 Identify peak shopping hours")
            
            # User preferences
            if st.session_state.user_preferences["favorite_categories"]:
                fav_cat = st.session_state.user_preferences["favorite_categories"][0]
                recommendations.append(f"💡 Deep dive into {fav_cat} customer demographics")
            
            # Default recommendations
            if not recommendations:
                recommendations = [
                    "💡 Explore customer lifetime value by segment",
                    "💡 Analyze payment method preferences",
                    "💡 Compare weekday vs weekend sales patterns"
                ]
            
            for rec in recommendations[:5]:
                st.markdown(rec)

# ================================================================
# 📊 Chart Gallery Page Display
# ================================================================

elif st.session_state.current_page == "charts":
    show_chart_gallery()

# ================================================================
# 📚 History Page Display
# ================================================================

elif st.session_state.current_page == "history":
    show_history_page()

# ================================================================
# 📱 Footer
# ================================================================

st.markdown("---")
footer_cols = st.columns([2, 1, 2])

with footer_cols[0]:
    st.caption("🚢 Maersk AI/ML Internship Assignment")

with footer_cols[1]:
    if st.button("ℹ️ Help", use_container_width=True):
        st.info("""
        **How to use:**
        
        1. **Ask naturally**: "What are the top categories?"
        2. **Follow up**: "Now show me trends for São Paulo"
        3. **Get insights**: "Tell me more about health_beauty"
        4. **Translate**: "Translate 'best sellers' to Portuguese"
        5. **Define**: "Define customer lifetime value"
        
        The AI remembers context and handles complex queries!
        """)

with footer_cols[2]:
    st.caption(f"💬 Messages: {len(st.session_state.chat_history)} | 🤖 AI-Powered Agent")

# ================================================================
# 🔧 System Status Indicator
# ================================================================

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    db_status = "🟢 Connected" if db_conn else "🔴 Disconnected"
    st.caption(f"Database: {db_status}")

with status_col2:
    ai_status = "🟢 Active" if model else "🔴 Offline"
    st.caption(f"AI Model: {ai_status}")

with status_col3:
    context_status = f"🟢 {len(st.session_state.chat_history)} msgs"
    st.caption(f"Context: {context_status}")

# ================================================================
# 🔄 Chart Click Handler
# ================================================================

# Handle chart click events from the chat interface
if st.session_state.get("chart_clicked") is not None:
    chart_index = st.session_state.chart_clicked
    st.session_state.selected_chart_index = chart_index
    st.session_state.current_page = "charts"
    st.session_state.chart_clicked = None
    st.rerun()