import streamlit as st
import pandas as pd
from langchain_community.llms import Ollama
from langchain_experimental.agents import create_pandas_dataframe_agent
import io

# Page configuration
st.set_page_config(
    page_title="Data Analysis Agent",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Title and description
st.title("📊 Data Analysis Agent with Ollama LLM")
st.markdown("Upload your CSV/Excel file and ask questions about your data using natural language!")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Ollama configuration
    ollama_url = st.secrets.get("OLLAMA_URL", "http://localhost:11434")
    
    model_name = "llama3.1:70B"
    temperature = 0.1
    
    st.divider()
    
    # File upload
    st.header("📁 Upload File")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls']
    )
    
    # Sheet selection for Excel files
    selected_sheet = None
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension in ['xlsx', 'xls']:
            # Read Excel file to get sheet names
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            if len(sheet_names) > 1:
                selected_sheet = st.selectbox(
                    "Select Sheet",
                    options=sheet_names,
                    help="Choose which sheet to analyze"
                )
            else:
                selected_sheet = sheet_names[0]
                st.info(f"Single sheet detected: {selected_sheet}")
    
    if st.button("🔄 Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Main content area
def load_data(file, file_type, sheet_name=None):
    """Load data from CSV or Excel file"""
    try:
        if file_type == 'csv':
            df = pd.read_csv(file)
        else:  # Excel
            df = pd.read_excel(file, sheet_name=sheet_name)
        return df, None
    except Exception as e:
        return None, str(e)

def create_agent(df, ollama_url, model_name, temperature):
    """Create the pandas dataframe agent with Ollama LLM"""
    try:
        # Initialize Ollama LLM
        llm = Ollama(
            base_url=ollama_url,
            model=model_name,
            temperature=temperature,
            num_ctx=4096  # Context window size
        )
        
        # Create the agent with updated parameters
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            allow_dangerous_code=True,
            handle_parsing_errors=True,
            max_iterations=20
        )
        
        return agent, None
    except Exception as e:
        return None, str(e)

# Load and display data
if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # Load data
    df, error = load_data(uploaded_file, file_extension, selected_sheet)
    
    if error:
        st.error(f"Error loading file: {error}")
    else:
        st.session_state.df = df
        
        # Display data info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Display dataframe
        st.subheader("📋 Data Preview")
        st.dataframe(df, use_container_width=True, height=300)
        
        # Data statistics
        with st.expander("📊 Data Statistics"):
            st.write(df.describe())
        
        with st.expander("ℹ️ Data Info"):
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
        
        # Create agent if not already created
        if st.session_state.agent is None:
            with st.spinner("Initializing AI Agent..."):
                agent, error = create_agent(
                    df, 
                    ollama_url, 
                    model_name, 
                    temperature
                )
                
                if error:
                    st.error(f"Error creating agent: {error}")
                    st.info("Please check if Ollama is running and accessible at the specified URL.")
                else:
                    st.session_state.agent = agent
                    st.success("✅ Agent is ready! You can now ask questions about your data.")

# Chat interface
if st.session_state.agent is not None:
    st.divider()
    st.subheader("💬 Chat with Your Data")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if question := st.chat_input("Ask a question about your data..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(question)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Add context from chat history
                    context = "\n".join([
                        f"{msg['role']}: {msg['content']}" 
                        for msg in st.session_state.chat_history[-5:]  # Last 5 messages
                    ])
                    
                    enhanced_question = f"""Based on the previous conversation context:
{context}

Current question: {question}

Please analyze the dataframe and provide a detailed answer."""
                    
                    response = st.session_state.agent.invoke(enhanced_question)
                    
                    # Extract output from response
                    if isinstance(response, dict):
                        answer = response.get('output', str(response))
                    else:
                        answer = str(response)
                    
                    st.markdown(answer)
                    
                    # Add assistant message to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": answer
                    })
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": error_msg
                    })

    # Quick question buttons
    st.markdown("### 🚀 Quick Questions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Show column names"):
            st.session_state.chat_history.append({
                "role": "user", 
                "content": "What are the column names?"
            })
            st.rerun()
    
    with col2:
        if st.button("📊 Data summary"):
            st.session_state.chat_history.append({
                "role": "user", 
                "content": "Give me a summary of this dataset"
            })
            st.rerun()
    
    with col3:
        if st.button("🔍 Missing values"):
            st.session_state.chat_history.append({
                "role": "user", 
                "content": "Are there any missing values?"
            })
            st.rerun()

else:
    # Instructions when no file is uploaded
    st.info("👈 Please upload a CSV or Excel file from the sidebar to get started!")
    
    st.markdown("""
    ### How to use:
    1. **Configure Ollama**: Enter the URL where Ollama is running (default: http://localhost:11434)
    2. **Upload File**: Choose your CSV or Excel file
    3. **Select Sheet** (for Excel): If multiple sheets exist, select the one to analyze
    4. **Ask Questions**: Use natural language to analyze your data
    
    ### Example Questions:
    - What are the column names in this dataset?
    - How many rows are there?
    - What is the average value of column X?
    - Show me the top 5 rows
    - Are there any missing values?
    - Create a correlation analysis
    - What are the unique values in column Y?
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Powered by Ollama (llama3.1:8b) & LangChain | Built with Streamlit</small>
</div>
""", unsafe_allow_html=True)