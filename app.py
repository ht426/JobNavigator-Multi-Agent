from typing import Callable, TypeVar
import os
import inspect
import shutil
import streamlit as st
import streamlit_analytics2 as streamlit_analytics
from dotenv import load_dotenv
from streamlit_chat import message
from streamlit_pills import pills
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from streamlit.delta_generator import DeltaGenerator
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.schema import HumanMessage
from custom_callback_handler import CustomStreamlitCallbackHandler
from agents import define_graph
# load_dotenv()

# ----------------- Set environment variables from Streamlit secrets or .env -----------------
# os.environ["LINKEDIN_EMAIL"] = st.secrets.get("LINKEDIN_EMAIL", "")
# os.environ["LINKEDIN_PASS"] = st.secrets.get("LINKEDIN_PASS", "")
# os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY", "")
# os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2") or st.secrets.get("LANGCHAIN_TRACING_V2", "")
# os.environ["LANGCHAIN_PROJECT"] = st.secrets.get("LANGCHAIN_PROJECT", "")
# os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", "")
# os.environ["SERPER_API_KEY"] = st.secrets.get("SERPER_API_KEY", "")
# os.environ["FIRECRAWL_API_KEY"] = st.secrets.get("FIRECRAWL_API_KEY", "")
# os.environ["LINKEDIN_SEARCH"] = st.secrets.get("LINKEDIN_JOB_SEARCH", "")
# os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "")
# os.environ["DEEPSEEK_API_KEY"] = st.secrets.get("DEEPSEEK_API_KEY", "")
# ------------------- 读取 secrets 并设置环境变量 -------------------
def load_secrets_to_env():
    """
    从 Streamlit secrets 读取所有 Key 并写入 os.environ 和 session_state
    """
    secrets_to_load = [
        "LINKEDIN_EMAIL",
        "LINKEDIN_PASS",
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_PROJECT",
        "GROQ_API_KEY",
        "SERPER_API_KEY",
        "FIRECRAWL_API_KEY",
        "LINKEDIN_JOB_SEARCH",
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
    ]

    for key in secrets_to_load:
        value = st.secrets.get(key, "")
        # 写入环境变量，如果环境变量已经存在，不覆盖
        if value and not os.environ.get(key):
            os.environ[key] = value
        # 写入 session_state，方便前端读取
        if key not in st.session_state:
            st.session_state[key] = value

# 调用一次，保证 secrets 生效
load_secrets_to_env()

# ----------------- Page configuration -----------------
st.set_page_config(layout="wide")
st.title("GenAI Career Assistant - 👨‍💼")
st.markdown("[Connect with me on LinkedIn](https://www.linkedin.com/in/aman-varyani-885725181/)")

streamlit_analytics.start_tracking()

# ----------------- Setup directories and dummy resume -----------------
temp_dir = "temp"
dummy_resume_path = os.path.abspath("dummy_resume.pdf")

if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

if not os.path.exists(dummy_resume_path):
    default_resume_path = "path/to/your/dummy_resume.pdf"
    shutil.copy(default_resume_path, dummy_resume_path)

# ----------------- Sidebar - File Upload -----------------
# uploaded_document = st.sidebar.file_uploader("Upload Your Resume", type="pdf")
#
# if not uploaded_document:
#     uploaded_document = open(dummy_resume_path, "rb")
#     st.sidebar.write("Using a dummy resume for demonstration purposes.")
#     st.sidebar.markdown(
#         f"[View Dummy Resume](https://drive.google.com/file/d/1vTdtIPXEjqGyVgUgCO6HLiG9TSPcJ5eM/view?usp=sharing)",
#         unsafe_allow_html=True
#     )
#
# bytes_data = uploaded_document.read()
# filepath = os.path.join(temp_dir, "resume.pdf")
# with open(filepath, "wb") as f:
#     f.write(bytes_data)
#
# st.markdown("**Resume uploaded successfully!**")

uploaded_document = st.sidebar.file_uploader("Upload Your Resume", type="pdf")

# 确保 temp 文件夹存在
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# 确定 resume 文件路径
filepath = os.path.join(temp_dir, "resume.pdf")

if uploaded_document is not None:
    # 用户上传了新文件 -> 覆盖 temp/resume.pdf
    bytes_data = uploaded_document.read()
    with open(filepath, "wb") as f:
        f.write(bytes_data)
    st.session_state["uploaded_resume_path"] = filepath
    st.markdown(f"**Resume uploaded successfully: {uploaded_document.name}**")
else:
    # 没上传文件
    # 1. 如果 session_state 有上次上传的文件，继续用它
    if "uploaded_resume_path" in st.session_state and os.path.exists(st.session_state["uploaded_resume_path"]):
        filepath = st.session_state["uploaded_resume_path"]
        st.sidebar.write("Using previously uploaded resume.")
    # 2. 否则使用 dummy 文件
    else:
        if not os.path.exists(dummy_resume_path):
            default_resume_path = "path/to/your/dummy_resume.pdf"
            shutil.copy(default_resume_path, dummy_resume_path)
        filepath = dummy_resume_path
        st.sidebar.write("Using a dummy resume for demonstration purposes.")
        st.sidebar.markdown(
            f"[View Dummy Resume](https://drive.google.com/file/d/1vTdtIPXEjqGyVgUgCO6HLiG9TSPcJ5eM/view?usp=sharing)",
            unsafe_allow_html=True
        )

# 将最终 resume 写入 temp/resume.pdf，保证分析器始终用最新文件
if filepath != os.path.join(temp_dir, "resume.pdf"):
    shutil.copy(filepath, os.path.join(temp_dir, "resume.pdf"))

# ----------------- Service Provider Selection -----------------
service_provider = "deepseek"  # 默认使用 Deepseek
user_choice = st.sidebar.selectbox(
    "Service Provider (Optional, override default Deepseek)",
    ("Keep Deepseek", "groq (llama-3.1-70b-versatile)", "openai"),
)
if user_choice != "Keep Deepseek":
    service_provider = user_choice

streamlit_analytics.stop_tracking()

# ----------------- Configure different models -----------------
def update_settings():
    """根据前端输入的 API Key 更新 settings"""
    global settings
    if service_provider == "deepseek":
        api_key = st.session_state.get("DEEPSEEK_API_KEY", "")
        model = st.session_state.get("deepseek_model_selected", "deepseek-chat")
        settings = {
            "model": model,
            "model_provider": "deepseek",
            "temperature": 0.3,
            "api_key": api_key
        }
    elif service_provider == "openai":
        api_key = st.session_state.get("OPENAI_API_KEY", "")
        model = st.session_state.get("openai_model_selected", "gpt-4o-mini")
        settings = {"model": model, "model_provider": "openai", "temperature": 0.3, "api_key": api_key}
    else:
        api_key = st.session_state.get("GROQ_API_KEY", "")
        settings = {"model": "llama-3.1-70b-versatile", "model_provider": "groq", "temperature": 0.3, "api_key": api_key}

# ----------------- Sidebar: API Key 输入 -----------------
if service_provider == "deepseek":
    if "deepseek_key_visible" not in st.session_state:
        st.session_state["deepseek_key_visible"] = False
    if st.sidebar.button("Enter Deepseek API Key (optional)"):
        st.session_state["deepseek_key_visible"] = True
    if st.session_state["deepseek_key_visible"]:
        api_key_deepseek = st.sidebar.text_input(
            "Deepseek API Key",
            st.session_state.get("DEEPSEEK_API_KEY", ""),
            type="password"
        )
        st.session_state["DEEPSEEK_API_KEY"] = api_key_deepseek
        os.environ["DEEPSEEK_API_KEY"] = api_key_deepseek

    if "deepseek_model_selected" not in st.session_state:
        st.session_state["deepseek_model_selected"] = "deepseek-chat"
    deepseek_model = st.sidebar.selectbox(
        "Select Deepseek Model",
        ("deepseek-chat", "deepseek-small"),
        index=["deepseek-chat", "deepseek-small"].index(
            st.session_state.get("deepseek_model_selected", "deepseek-chat"))
    )
    st.session_state["deepseek_model_selected"] = deepseek_model

elif service_provider == "openai":
    if "openai_key_visible" not in st.session_state:
        st.session_state["openai_key_visible"] = False
    if st.sidebar.button("Enter OpenAI API Key (optional)"):
        st.session_state["openai_key_visible"] = True
    if st.session_state["openai_key_visible"]:
        api_key_openai = st.sidebar.text_input(
            "OpenAI API Key",
            st.session_state.get("OPENAI_API_KEY", ""),
            type="password"
        )
        st.session_state["OPENAI_API_KEY"] = api_key_openai
        os.environ["OPENAI_API_KEY"] = api_key_openai

    if "openai_model_selected" not in st.session_state:
        st.session_state["openai_model_selected"] = "gpt-4o-mini"
    openai_model = st.sidebar.selectbox(
        "OpenAI Model",
        ("gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"),
        index=0
    )
    st.session_state["openai_model_selected"] = openai_model

else:
    if "groq_key_visible" not in st.session_state:
        st.session_state["groq_key_visible"] = False
    if st.sidebar.button("Enter Groq API Key (optional)"):
        st.session_state["groq_key_visible"] = True
    if st.session_state["groq_key_visible"]:
        api_key_groq = st.sidebar.text_input(
            "Groq API Key",
            st.session_state.get("GROQ_API_KEY", ""),
            type="password"
        )
        st.session_state["GROQ_API_KEY"] = api_key_groq
        os.environ["GROQ_API_KEY"] = api_key_groq

update_settings()  # 确保 settings 已更新

# ----------------- Sidebar Notes -----------------
st.sidebar.markdown("""
**Note:** \n
This multi-agent system works best with Deepseek by default.\n
You can override with OpenAI or Groq. Any key provided will only be used in this session.
""")
st.sidebar.markdown("""
<div style="padding:10px 0;">
    If you like the project, give a 
    <a href="https://github.com/ht426" target="_blank" style="text-decoration:none;">
        ⭐ on GitHub
    </a>
</div>
""", unsafe_allow_html=True)

# ----------------- Initialize flow and message history -----------------
flow_graph = define_graph()
message_history = StreamlitChatMessageHistory()

for key, default in [("active_option_index", None), ("interaction_history", []),
                     ("response_history", ["Hello! How can I assist you today?"]),
                     ("user_query_history", ["Hi there! 👋"])]:
    if key not in st.session_state:
        st.session_state[key] = default

conversation_container = st.container()
input_section = st.container()

# ----------------- Functions -----------------
def initialize_callback_handler(main_container: DeltaGenerator):
    V = TypeVar("V")
    def wrap_function(func: Callable[..., V]) -> Callable[..., V]:
        context = get_script_run_ctx()
        def wrapped(*args, **kwargs) -> V:
            add_script_run_ctx(ctx=context)
            return func(*args, **kwargs)
        return wrapped

    streamlit_callback_instance = CustomStreamlitCallbackHandler(parent_container=main_container)
    for method_name, method in inspect.getmembers(streamlit_callback_instance, predicate=inspect.ismethod):
        setattr(streamlit_callback_instance, method_name, wrap_function(method))
    return streamlit_callback_instance


def execute_chat_conversation(user_input, graph):
    callback_handler_instance = initialize_callback_handler(st.container())
    callback_handler = callback_handler_instance

    messages_list = list(message_history.messages) + [HumanMessage(content=user_input)]

    update_settings()

    try:
        output = graph.invoke(
            {
                "messages": messages_list,
                "user_input": user_input,
                "config": settings,
                "callback": callback_handler,
            },
            {"recursion_limit": 30},
        )
        message_output = output.get("messages")[-1]
        message_history.clear()
        message_history.add_messages(output.get("messages"))

    except Exception as exc:
        st.error(f"Error occurred: {exc}")
        return ":( Sorry, Some error occurred. Can you please try again?"

    return message_output.content

# ----------------- Clear Chat -----------------
if st.button("Clear Chat"):
    st.session_state["user_query_history"] = []
    st.session_state["response_history"] = []
    message_history.clear()
    st.rerun()

# ----------------- Chat Interface -----------------
streamlit_analytics.start_tracking()

with input_section:
    options = [
        "Identify top trends in the tech industry relevant to gen ai",
        "Find emerging technologies and their potential impact on job opportunities",
        "Summarize my resume",
        "Create a career path visualization based on my skills and interests from my resume",
        "GenAI Jobs at Microsoft",
        "Job Search GenAI jobs in India.",
        "Analyze my resume and suggest a suitable job role and search for relevant job listings",
        "Generate a cover letter for my resume.",
    ]
    icons = ["🔍", "🌐", "📝", "📈", "💼", "🌟", "✉️", "🧠"]

    selected_query = pills(
        "Pick a question for query:",
        options,
        clearable=None,
        icons=icons,
        index=st.session_state["active_option_index"],
        key="pills",
    )
    if selected_query:
        st.session_state["active_option_index"] = options.index(selected_query)

    with st.form(key="query_form", clear_on_submit=True):
        user_input_query = st.text_input(
            "Query:",
            value=(selected_query if selected_query else "Detail analysis of latest layoff news India?"),
            placeholder="📝 Write your query or select from the above",
            key="input",
        )
        submit_query_button = st.form_submit_button(label="Send")

    if submit_query_button:
        if not uploaded_document:
            st.error("Please upload your resume before submitting a query.")
        elif service_provider == "openai" and not st.session_state.get("OPENAI_API_KEY"):
            st.error("Please enter your API key before submitting a query.")
        elif service_provider == "deepseek" and not st.session_state.get("DEEPSEEK_API_KEY"):
            st.error("Please enter your API key before submitting a query.")
        elif service_provider.startswith("groq") and not st.session_state.get("GROQ_API_KEY", ""):
            st.error("Please enter your API key before submitting a query.")
        elif user_input_query:
            chat_output = execute_chat_conversation(user_input_query, flow_graph)
            st.session_state["user_query_history"].append(user_input_query)
            st.session_state["response_history"].append(chat_output)
            st.session_state["last_input"] = user_input_query
            st.session_state["active_option_index"] = None

# ----------------- Display Chat History -----------------
if st.session_state["response_history"]:
    with conversation_container:
        for i in range(len(st.session_state["response_history"])):
            message(
                st.session_state["user_query_history"][i],
                is_user=True,
                key=str(i) + "_user",
                avatar_style="fun-emoji",
            )
            message(
                st.session_state["response_history"][i],
                key=str(i),
                avatar_style="bottts",
            )

streamlit_analytics.stop_tracking()
