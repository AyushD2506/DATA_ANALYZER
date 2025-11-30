import streamlit as st
import pandas as pd
from langchain_community.llms import Ollama
from langchain_experimental.agents import create_pandas_dataframe_agent
import io
import traceback

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
if 'show_debug' not in st.session_state:
    st.session_state.show_debug = False

# Title and description
st.title("📊 Data Analysis Agent with Ollama LLM")
st.markdown("Upload your CSV/Excel file and ask questions about your data using natural language!")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Ollama configuration - ALWAYS keep secret, never display
    ollama_url = st.secrets.get("OLLAMA_URL", "http://localhost:11434")
    
    # Model configuration (visible to user)
    # st.info(f"🤖 Model: llama3.1:70B")
    model_name = "llama3.1:70B"
    temperature = 0.1
    
    st.divider()
    
    # Debug mode toggle
    st.session_state.show_debug =False
    
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
            try:
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
            except Exception as e:
                st.error(f"Error reading Excel file: {str(e)}")
                if st.session_state.show_debug:
                    st.code(traceback.format_exc())
    
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
        error_details = {
            'message': str(e),
            'type': type(e).__name__,
            'traceback': traceback.format_exc()
        }
        return None, error_details

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
        error_details = {
            'message': str(e),
            'type': type(e).__name__,
            'traceback': traceback.format_exc()
        }
        return None, error_details

def display_error(error_details, context=""):
    """Display error with appropriate level of detail"""
    if isinstance(error_details, dict):
        error_msg = error_details.get('message', 'Unknown error')
        error_type = error_details.get('type', 'Error')
        error_trace = error_details.get('traceback', '')
    else:
        error_msg = str(error_details)
        error_type = 'Error'
        error_trace = ''
    
    # Always show basic error
    st.error(f"**{error_type}**: {error_msg}")
    
    # Add context-specific help
    if context:
        st.warning(f"**Context**: {context}")
    
    # Show detailed debug info if enabled
    if st.session_state.show_debug and error_trace:
        with st.expander("🔍 View Full Error Details"):
            st.code(error_trace, language="python")
    
    # Provide helpful suggestions based on error type
    suggestions = []
    if "connection" in error_msg.lower() or "refused" in error_msg.lower():
        suggestions.append("- Check if Ollama service is running: `ollama serve`")
        suggestions.append("- Verify that the Ollama service is accessible")
        suggestions.append("- Check your network connection")
    elif "model" in error_msg.lower():
        suggestions.append(f"- Check if model '{model_name}' is installed: `ollama list`")
        suggestions.append(f"- Pull the model if needed: `ollama pull {model_name}`")
    elif "memory" in error_msg.lower() or "ram" in error_msg.lower():
        suggestions.append("- Try a smaller model (e.g., llama3.1:8b instead of 70b)")
        suggestions.append("- Close other applications to free up memory")
    elif "parsing" in error_msg.lower():
        suggestions.append("- The AI model's response format was unexpected")
        suggestions.append("- Try rephrasing your question more simply")
    
    if suggestions:
        st.info("**💡 Suggestions:**\n" + "\n".join(suggestions))

# Load and display data
if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # Load data
    df, error = load_data(uploaded_file, file_extension, selected_sheet)
    
    if error:
        display_error(error, f"Failed to load {file_extension.upper()} file")
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
                    display_error(error, "Failed to initialize AI agent")
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
            if message["role"] == "assistant" and message.get("is_error"):
                st.error(message["content"])
                if st.session_state.show_debug and "debug_info" in message:
                    with st.expander("🔍 Error Details"):
                        st.code(message["debug_info"])
            else:
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
                        if not msg.get("is_error")
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
                        "content": answer,
                        "is_error": False
                    })
                    
                except Exception as e:
                    error_msg = f"**Error processing your question**\n\n{type(e).__name__}: {str(e)}"
                    error_trace = traceback.format_exc()
                    
                    # Display error
                    st.error(error_msg)
                    
                    # Show debug info if enabled
                    if st.session_state.show_debug:
                        with st.expander("🔍 View Full Error Details"):
                            st.code(error_trace, language="python")
                    
                    # Provide suggestions
                    suggestions = []
                    if "timeout" in str(e).lower():
                        suggestions.append("- The query took too long. Try a simpler question.")
                        suggestions.append("- The model might be overloaded.")
                    elif "parsing" in str(e).lower():
                        suggestions.append("- Try rephrasing your question more clearly.")
                        suggestions.append("- Ask for specific information rather than complex analysis.")
                    elif "column" in str(e).lower() or "key" in str(e).lower():
                        suggestions.append("- Check that the column names in your question match the dataset.")
                        suggestions.append("- Use the 'Show column names' button to see available columns.")
                    
                    if suggestions:
                        st.info("**💡 Suggestions:**\n" + "\n".join(suggestions))
                    
                    # Add error to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": error_msg,
                        "is_error": True,
                        "debug_info": error_trace if st.session_state.show_debug else None
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
    1. **Configure Ollama**: Ensure Ollama service is running and configured in secrets
    2. **Enable Debug Mode** (optional): Toggle in sidebar to see detailed error messages
    3. **Upload File**: Choose your CSV or Excel file
    4. **Select Sheet** (for Excel): If multiple sheets exist, select the one to analyze
    5. **Ask Questions**: Use natural language to analyze your data
    
    ### Example Questions:
    - What are the column names in this dataset?
    - How many rows are there?
    - What is the average value of column X?
    - Show me the top 5 rows
    - Are there any missing values?
    - Create a correlation analysis
    - What are the unique values in column Y?
    
    ### Troubleshooting:
    - **Enable Debug Mode** in the sidebar to see full error details
    - Check that Ollama is running: `ollama serve`
    - Verify your model is installed: `ollama list`
    - Pull the model if needed: `ollama pull llama3.1:70b`
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Powered by Ollama (llama3.1:70b) & LangChain | Built with Streamlit</small>
</div>
""", unsafe_allow_html=True)