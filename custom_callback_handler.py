from typing import Any
from langchain_community.callbacks import StreamlitCallbackHandler
from streamlit.external.langchain.streamlit_callback_handler import (
    StreamlitCallbackHandler,
    LLMThought,
)
from langchain.schema import AgentAction

# LangChain 提供的 Streamlit 回调类，用于在 Streamlit 中实时展示 Agent 执行日志。
class CustomStreamlitCallbackHandler(StreamlitCallbackHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_sequence = []  # 记录agent执行顺序

    def write_agent_name(self, name: str):
        self._parent_container.write(name)
        # 记录agent执行顺序
        self.agent_sequence.append(name)

    def write_output(self, text: str):
        """在 Streamlit 界面输出日志或调试信息"""
        # 你可以根据风格选择 markdown 或普通 write
        self._parent_container.markdown(
            f"<div style='color:gray; font-size:0.9em;'>{text}</div>",
            unsafe_allow_html=True,
        )

    def get_agent_sequence(self):
        return self.agent_sequence

    def clear_agent_sequence(self):
        self.agent_sequence = []

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        # 显示agent正在执行的动作
        tool_name = action.tool
        tool_input = action.tool_input
        self._parent_container.write(f"🔧 正在执行: {tool_name}")
        return super().on_agent_action(action, **kwargs)


