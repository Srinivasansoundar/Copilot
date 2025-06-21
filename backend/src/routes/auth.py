from datetime import datetime, timedelta
from fastapi.responses import FileResponse
from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from dotenv import load_dotenv
from pandasai import SmartDataframe
from langchain_groq import ChatGroq
import pandas as pd
import os
import json
import logging
from src.schemas.user import UserCreate, Token
from src.database import get_db
from src.models.user import User
import re
import  uuid
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import matplotlib.dates as mdates
from src.utils.auth import (
    authenticate_user,
    create_access_token,
    get_hashed_password,
    get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Load environment variables
load_dotenv()
DIAGRAMS_DIR = Path("./diagrams")
DIAGRAMS_DIR.mkdir(exist_ok=True)
router = APIRouter(prefix='/api', tags=['auth_and_chat'])

# ---------------- LLM + SmartDataFrame Setup ----------------
DATASET_PATH = "C:\\Users\\Srinivasan\\Documents\\skills\\Projects\\eshipz\\csv-query-assistant\\backend\\filename.csv"
df = pd.read_csv(DATASET_PATH)

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("Missing GROQ_API_KEY in .env")

llm = ChatGroq(
    groq_api_key=api_key.strip(),
    model_name="llama3-70b-8192",
    temperature=0.2
)

smart_df = SmartDataframe(df, config={
    "llm": llm,
    "enable_cache": False,
    "save_logs": False,
    "save_charts": False,
    "verbose": True
})

# ---------------- Register ----------------
@router.post("/auth/register")
async def register_user(user: UserCreate, db: AsyncSession = Depends(get_db)):
    if (await db.execute(select(User).filter(User.username == user.username))).scalars().first():
        raise HTTPException(status_code=400, detail="Username already registered")
    if (await db.execute(select(User).filter(User.email == user.email))).scalars().first():
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = get_hashed_password(user.password)
    db_user = User(email=user.email, username=user.username, hashed_password=hashed_password)
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user

# ---------------- Login ----------------
@router.post("/auth/login", response_model=Token)
async def login_for_access_token(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.id},
        expires_delta=access_token_expires
    )

    redis = request.app.state.redis
    await redis.delete(f"user:{user.id}:conversation")
    await redis.hset(f"user:{user.id}:context", mapping={
        "session_start": datetime.utcnow().isoformat(),
        "total_queries": "0",
        "last_activity": datetime.utcnow().isoformat()
    })
    await redis.expire(f"user:{user.id}:context", 86400)
    return Token(access_token=access_token, token_type="Bearer")

# ---------------- Chat ----------------
@router.post("/chat/ask")
async def ask_bot(
    request: Request,
    payload: dict,
    current_user: User = Depends(get_current_user)  # Extracted from JWT
):
    redis = request.app.state.redis
    question = payload.get("query")

    if not question:
        return {"response": {"answer": "Missing question"}}

    user_id = str(current_user.id)
    conversation_key = f"user:{user_id}:conversation"
    context_key = f"user:{user_id}:context"

    # Store the user message first
    await redis.rpush(conversation_key, json.dumps({"role": "user", "message": question}))

    # -------- Fetch Last N Messages --------
    MAX_HISTORY_TURNS = 6  # 3 user-bot pairs
    history_raw = await redis.lrange(conversation_key, -MAX_HISTORY_TURNS, -1)
    history = [json.loads(msg) for msg in history_raw]

    # -------- Format History as Conversation --------
    formatted_history = ""
    for msg in history:
        role = msg.get("role", "user").capitalize()
        message = msg.get("message", "")
        formatted_history += f"{role}: {message}\n"

    try:
        # Determine if visualization is requested
        visualization_keywords = ["diagram", "chart", "plot", "graph", "visualize", 
                                "visualization", "show me", "display"]
        visualization_requested = any(word in question.lower() for word in visualization_keywords)
        
        logger.info(f"Visualization requested: {visualization_requested}")
       
        # Get the LLM response
        try:
            llm_response = await get_llm_response(formatted_history, question, visualization_requested)
        except Exception as e:
            logger.error(f"Error in get_llm_response: {e}", exc_info=True)
            raise
        
        # Ensure the response is a string
        if not isinstance(llm_response.get("answer"), str):
            llm_response["answer"] = str(llm_response["answer"])
        
        logger.info(f"Generated response: {llm_response['answer']}")
        
        visualization_data = None
        if visualization_requested:
            try:
                # Generate diagram with extracted parameters
                diagram_metadata = await generate_diagram2(
                    smart_df, 
                    question, 
                    llm_response.get("viz_params", {}),
                )
                
                if "error" in diagram_metadata:
                    logger.warning(f"Diagram generation error: {diagram_metadata['error']}")
                    visualization_data = {
                        "status": "error",
                        "error_message": diagram_metadata['error'],
                        "fallback_path": f"/api/diagram/{os.path.basename(diagram_metadata.get('path', ''))}"
                    }
                else:
                    logger.info(f"Generated diagram metadata: {diagram_metadata}")
                    visualization_data = {
                        "status": "success",
                        "diagram_path": f"/api/diagram/{os.path.basename(diagram_metadata.get('path', ''))}",
                        "type": diagram_metadata.get("type", "unknown"),
                        "title": diagram_metadata.get("title", "Untitled"),
                        "x_axis": diagram_metadata.get("x_column", ""),
                        "y_axis": diagram_metadata.get("y_column", ""),
                    }
            except Exception as e:
                logger.error(f"Error in generate_diagram2: {e}", exc_info=True)
                # Don't raise here, continue with text response
                visualization_data = {
                    "status": "error",
                    "error_message": f"Visualization generation failed: {str(e)}"
                }
        
        # Return the response with structured format
        result = {
            "status": "success",
            "response": {
                "answer": llm_response["answer"]
            },
        }
        
        if visualization_data:
            result["visualization"] = visualization_data

        # Store bot response
        await redis.rpush(conversation_key, json.dumps({"role": "bot", "message": result['response']['answer']}))

        # Update context metadata
        await redis.hincrby(context_key, "total_queries", 1)
        await redis.hset(context_key, "last_activity", datetime.utcnow().isoformat())

        return result
    
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def get_llm_response(history, query, visualization_requested):
    """Get response from LLM with visualization parameters if needed"""
    if visualization_requested:
        viz_prompt = f"""
You are a logistics data analysis assistant. You will be provided with logistics CSV data and a conversation history.

Your tasks are:

### Tasks:
1. Answer the current question: "{query}" using insights from the data.
2. Determine the most suitable chart type to visualize the answer.
3. Recommend the visualization parameters.
4. **DO NOT** generate or include any diagram, plot, or code.
5. Return your output strictly in JSON format as shown below.
6. This is the conversation history: {history}

### IMPORTANT COLUMN SELECTION RULE:
You MUST only select x_column and y_column from the following exact list of column names:
'_id', 'date', 'is_to_pay', 'is_reverse', 'creation_date', 'order_id',
'order_source', 'parcel_contents', 'service_type', 'service_options',
'account_info', 'order_details', 'parcels', 'awb', 'label_meta',
'charge_weight', 'total_charge', 'slug', 'package_count', 'purpose',
'is_cod', 'order_status', 'shipment_type', 'customer_referenc',
'entered_weight', 'invoice_details', 'tracking_link', 'meta_collection',
'tracking_status', 'active', 'labels_downloaded', 'async_ops_completed',
'label_format', 'pickup_meta', 'gst_invoices', 'vendor_name',
'is_csb_v_mode', 'delivery_attempts', 'first_ofd_date',
'latest_checkpoint_date', 'latest_ofd_date', 'tracking_actual_delivery',
'tracking_latest_message', 'tracking_latest_msg', 'tracking_pick_date',
'tracking_return_pick_date', 'tracking_sub_status',
'webhook_triggered_hash', 'webhook_triggered_states',
'webhook_triggered_timestamps', 'tracking_expected_delivery',
'pod_link'

### VALIDATION RULES BEFORE RESPONDING:
- Confirm both x_column and y_column exist in the schema above.
- Ensure agg_function makes sense for the columns selected.
- Ensure chart_type is suitable for logistics data analytics.
- Most importantly, do not return any Python code or charts — just return structured insight.

### REQUIRED OUTPUT FORMAT (PandasAI compatible):
Return the result using the following format:
{{"type": "string", "value": json.dumps(result)}}  
Where `result` is:
{{
    "answer": "Your detailed answer here with key insights from the logistics data",
    "visualization": {{
        "chart_type": "bar|line|pie|scatter|histogram|heatmap",
        "x_column": "Must match exact column name from schema",
        "y_column": "Must match exact column name from schema",
        "agg_function": "count|sum|mean|median",
        "group_by": "Valid column name or null",
        "title": "Descriptive chart title for logistics data",
        "limit": 10
    }}
}}

REMINDER: Do NOT return any code. Return ONLY the structured result in JSON format as shown above.
"""
   
      
        response_text = smart_df.chat(viz_prompt)
        logger.info(response_text)
        # Handle case where smart_df.chat returns a dict
        if isinstance(response_text, dict):
            response_text = response_text.get('value', str(response_text))
        
        try:
            logger.info("Parsing visualization response")
            
            # Handle PandasAI response format
            if isinstance(response_text, dict):
                if "value" in response_text:
                    # Extract the JSON string from PandasAI format
                    response_json = json.loads(response_text["value"])
                else:
                    response_json = response_text
            else:
                # Handle string responses with JSON blocks
                json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    response_json = json.loads(json_match.group(1))
                else:
                    # Try parsing the whole response as JSON
                    response_json = json.loads(response_text)
                
            answer = response_json.get("answer", "")
            viz_params = response_json.get("visualization", {})
            
            # Map JSON fields to expected viz param names
            if "chart_type" in viz_params:
                viz_params["diagram_type"] = viz_params.pop("chart_type")
            
            return {
                "answer": answer,
                "viz_params": viz_params       
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}")
            # Fallback: extract answer from text response
            return {
                "answer": response_text,
                "viz_params": {
                    "diagram_type": "auto",
                    "x_column": None,
                    "y_column": None,
                    "agg_function": "count",
                    "group_by": None,
                    "title": None,
                    "limit": 10
                }
            }
    else:
        answer_prompt = f"""
        You are a data analysis assistant. Use the provided data and conversation history to answer the current question accurately and concisely.

        ### Conversation History:
        {history}  

        ### Available Data Schema:
        {smart_df.dtypes}  

        ### Current Question:
        {query}

        ### Instructions:
        Provide a clear, concise answer to the question based on the available data.
        
        IMPORTANT: Return in PandasAI format: {{"type": "string", "value": json.dumps({{"answer": "Your answer here"}})}}
        
        Example:
        import json
        result = {{"answer": "Top carrier by shipment volume is Delhivery with 1,234 shipments."}}
        return {{"type": "string", "value": json.dumps(result)}}
        """

        response = smart_df.chat(answer_prompt)
        logger.info(response)
        # # Handle case where smart_df.chat returns a dict
        # if isinstance(response, dict):
        #     response = response.get('value', str(response))
            
        # logger.info(f"Non-viz response: {response}")
        
        # # Try to parse as JSON
        # try:
        #     # Handle PandasAI response format
        #     if isinstance(response, dict):
        #         if "value" in response:
        #             # Extract the JSON string from PandasAI format
        #             response_json = json.loads(response["value"])
        #             answer = response_json.get("answer", "")
        #             return {"answer": answer}
        #         else:
        #             answer = response.get("answer", str(response))
        #             return {"answer": answer}
            
        #     # Handle string responses
        #     if isinstance(response, str):
        #         # Try extracting JSON from markdown code block
        #         json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
        #         if json_match:
        #             try:
        #                 response_json = json.loads(json_match.group(1))
        #                 answer = response_json.get("answer", "")
        #                 return {"answer": answer}
        #             except json.JSONDecodeError:
        #                 logger.warning("Found code block but couldn't parse as JSON")
                
        #         # Try parsing the entire response as JSON
        #         try:
        #             response_json = json.loads(response)
        #             answer = response_json.get("answer", "")
        #             return {"answer": answer}
        #         except json.JSONDecodeError:
        #             logger.warning("Response is not valid JSON")
    
        # except Exception as e:
        #     logger.error(f"Error parsing response: {str(e)}")
    
        # Fallback: Return the raw response as answer
        return {"answer": str(response)}


async def generate_diagram2(df, query=None, viz_params=None, time_related_query=False):
    """Generate diagram with time-series optimization if needed"""
    # Use empty dict if viz_params is None
    viz_params = viz_params or {}
    
    # Get diagram parameters
    diagram_type = viz_params.get("diagram_type", "auto")
    x_column = viz_params.get("x_column")
    y_column = viz_params.get("y_column")
    agg_function = viz_params.get("agg_function", "count")
    group_by = viz_params.get("group_by")
    title = viz_params.get("title")
    
    # Parse limit with fallback to default
    try:
        limit = int(viz_params.get("limit", 10))
    except (ValueError, TypeError):
        limit = 10
    
    # Generate the diagram directly using the enhanced function
    return generate_enhanced_diagram(
        df=df,
        diagram_type=diagram_type,
        x_column=x_column,
        y_column=y_column,
        agg_function=agg_function,
        group_by=group_by,
        title=title,
        limit=limit
    )


def prepare_data(df, diagram_type, x_column, y_column, group_by, agg_function, limit):
    """Prepare and validate data for plotting."""
    if group_by:
        if agg_function == "count":
            plot_data = df.groupby(group_by).size().reset_index(name='count')
        else:
            if not y_column:
                raise ValueError("y_column required when using aggregation function")
            plot_data = df.groupby(group_by).agg({y_column: agg_function}).reset_index()
        
        if diagram_type in ["bar", "pie"]:
            sort_col = y_column if agg_function != "count" else "count"
            plot_data = plot_data.sort_values(sort_col, ascending=False).head(limit)
    else:
        plot_data = df.copy()
        
    prepared_data = plot_data.copy()
    
    # Type conversion for numeric plots
    if diagram_type in ["scatter", "line"]:
        def safe_numeric_conversion(series):
            """
            Attempt multiple strategies to convert to numeric
            
            Strategies (in order):
            1. Direct numeric conversion
            2. Remove non-numeric characters
            3. Extract numeric part
            """
            # Strategy 1: Direct conversion
            converted = pd.to_numeric(series, errors='coerce')
            
            # If direct conversion fails, try more aggressive approaches
            if converted.isna().all():
                # Strategy 2: Remove non-numeric characters
                converted = pd.to_numeric(
                    series.astype(str).str.replace(r'[^0-9.]', '', regex=True), 
                    errors='coerce'
                )
            
            return converted
        
        # Apply safe conversion to both columns
        if x_column and x_column in prepared_data.columns:
            prepared_data[x_column] = safe_numeric_conversion(prepared_data[x_column])
        
        if y_column and y_column in prepared_data.columns:
            prepared_data[y_column] = safe_numeric_conversion(prepared_data[y_column])
        
        # Drop rows where either column is non-numeric
        prepared_data = prepared_data.dropna(subset=[col for col in [x_column, y_column] if col])
    
    return prepared_data


def generate_histogram(data, x_column):
    """Generate histogram plot."""
    if not x_column:
        raise ValueError("x_column required for histogram")
    plt.hist(data[x_column].dropna(), bins=20, alpha=0.7)
    plt.xlabel(x_column)
    plt.ylabel("Frequency")


def generate_scatter(data, x_column, y_column):
    """Generate scatter plot with trendline."""
    if not (x_column and y_column):
        raise ValueError("Both x_column and y_column required for scatter plot")
    
    plt.scatter(data[x_column], data[y_column], alpha=0.7)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    
    # Add trendline if sufficient data points
    if len(data) > 1:
        try:
            z = np.polyfit(data[x_column], data[y_column], 1)
            p = np.poly1d(z)
            plt.plot(data[x_column], p(data[x_column]), "r--", 
                    alpha=0.8, label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")
            plt.legend()
        except Exception as e:
            logger.warning(f"Trendline failed: {str(e)}")


def generate_barplot(data, x_column, y_column, group_by, agg_function, limit, categorical_cols):
    """Generate bar plot with proper categorical handling."""
    if group_by:
        y_col = y_column if y_column else 'count'
        sns.barplot(x=group_by, y=y_col, data=data, palette='viridis')
        plt.xlabel(group_by)
        plt.ylabel(y_col)
    else:
        if x_column in categorical_cols:
            counts = data[x_column].value_counts().head(limit)
            sns.barplot(x=counts.index, y=counts.values, palette='viridis')
            plt.xlabel(x_column)
            plt.ylabel("Count")
        else:
            plt.bar(range(len(data[x_column].head(limit))), data[x_column].head(limit))
            plt.xlabel("Index")
            plt.ylabel(x_column)
    
    if x_column in categorical_cols and len(data[x_column].unique()) > 5:
        plt.xticks(rotation=45, ha='right')


def generate_lineplot(data, x_column, y_column, datetime_cols, numeric_cols):
    """Generate line plot with datetime support."""
    if not (x_column and y_column):
        raise ValueError("Both x_column and y_column required for line plot")
    
    if x_column in datetime_cols:
        data = data.sort_values(x_column)
        plt.plot(data[x_column], data[y_column], marker='o', markersize=4)
        plt.gcf().autofmt_xdate()
        if data[x_column].dt.minute.nunique() > 1:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        else:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    elif x_column in numeric_cols:
        data = data.sort_values(x_column)
        plt.plot(data[x_column], data[y_column], marker='o', markersize=4)
    else:
        plt.plot(data[x_column].head(50), data[y_column].head(50), marker='o', markersize=4)
    
    plt.xlabel(x_column)
    plt.ylabel(y_column)


def generate_piechart(data, x_column, categorical_cols, limit):
    """Generate pie chart with automatic type handling and limit support."""
    if not x_column:
        raise ValueError("x_column required for pie chart")

    # Count values based on column type
    if data[x_column].dtype == 'bool':
        counts = data[x_column].map({True: "True", False: "False"}).value_counts()
    elif np.issubdtype(data[x_column].dtype, np.number):
        counts = pd.cut(data[x_column], bins=5).value_counts()
    elif x_column in categorical_cols:
        counts = data[x_column].astype(str).value_counts()
    else:
        raise ValueError("Pie charts require categorical or binned numeric data")

    # Limit categories
    if limit and len(counts) > limit:
        top_counts = counts.iloc[:limit]
        others_sum = counts.iloc[limit:].sum()
        if others_sum > 0:
            counts = pd.concat([top_counts, pd.Series({'Others': others_sum})])
        else:
            counts = top_counts

    # Plot
    plt.pie(
        counts, 
        labels=counts.index, 
        autopct='%1.1f%%', 
        shadow=True, 
        startangle=90, 
        colors=plt.cm.viridis(np.linspace(0, 1, len(counts)))
    )
    plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle.


def generate_error_diagram(error_message):
    """Generate an error diagram with the error message."""
    filename = f"diagram_error_{uuid.uuid4()}.png"
    filepath = DIAGRAMS_DIR / filename
    
    plt.figure(figsize=(8, 6))
    plt.text(0.5, 0.5, f"Could not generate diagram:\n{error_message}", 
             horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes, fontsize=12, wrap=True)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    return str(filepath)


def generate_enhanced_diagram(df, diagram_type="auto", x_column=None, y_column=None, 
                             agg_function="count", group_by=None, title=None, limit=10):
    """Generate an enhanced matplotlib diagram with robust data handling."""
    filename = f"diagram_{uuid.uuid4()}.png"
    filepath = DIAGRAMS_DIR / filename
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    
    # Data type detection
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    try:
        # Data preparation
        plot_data = prepare_data(df, diagram_type, x_column, y_column, group_by, agg_function, limit)
        
        # Plot generation
        if diagram_type == "histogram":
            generate_histogram(plot_data, x_column)
            
        elif diagram_type == "scatter":
            generate_scatter(plot_data, x_column, y_column)
            
        elif diagram_type == "bar":
            generate_barplot(plot_data, x_column, y_column, group_by, agg_function, limit, categorical_cols)
            
        elif diagram_type == "line":
            generate_lineplot(plot_data, x_column, y_column, datetime_cols, numeric_cols)
            
        elif diagram_type == "pie":
            generate_piechart(plot_data, x_column, categorical_cols, limit)
            
        else:
            raise ValueError(f"Unsupported diagram type: {diagram_type}")

        # Final touches
        if title:
            plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            "path": str(filepath),
            "type": diagram_type,
            "x_column": x_column,
            "y_column": y_column,
            "title": title
        }
        
    except Exception as e:
        logger.error(f"Diagram generation failed: {str(e)}", exc_info=True)
        plt.close()  # Ensure plot is closed even on error
        return {
            "path": str(generate_error_diagram(str(e))),
            "error": str(e)
        }


@router.get("/diagram/{filename}")
async def get_diagram(filename: str):
    """Serve a generated diagram"""
    filepath = DIAGRAMS_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Diagram not found")
    return FileResponse(filepath)
# ---------------- Logout ----------------
@router.post("/auth/logout")
async def logout_user(request: Request, current_user: User = Depends(get_current_user)):
    redis = request.app.state.redis
    await redis.delete(f"user:{current_user.id}:conversation")
    await redis.delete(f"user:{current_user.id}:context")
    return {"message": "Successfully logged out"}
