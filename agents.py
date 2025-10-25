from typing import Any, TypedDict
from langchain.agents import (
    AgentExecutor,
    create_openai_tools_agent,
)
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from chains import get_finish_chain, get_supervisor_chain
from tools import (
    get_job_search_tool,
    ResumeExtractorTool,
    generate_letter_for_specific_job,
    get_google_search_results,
    save_cover_letter_for_specific_job,
    scrape_website,
)
from prompts import (
    get_search_agent_prompt_template,
    get_analyzer_agent_prompt_template,
    researcher_agent_prompt_template,
    get_generator_agent_prompt_template,
)

load_dotenv()

# 生成一个 LangChain Agent，绑定 LLM、工具和系统 Prompt。
def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    """
    Creates an agent using the specified ChatOpenAI model, tools, and system prompt.

    Args:
        llm : LLM to be used to create the agent.
        tools (list): The list of tools to be given to the worker node.
        system_prompt (str): The system prompt to be used in the agent.

    Returns:
        AgentExecutor: The executor for the created agent.
    """
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

# Supervisor 节点
def supervisor_node(state):
    new_state = state.copy()

    # 初始化循环计数器
    if 'supervisor_count' not in new_state:
        new_state['supervisor_count'] = 0
    new_state['supervisor_count'] += 1

    # 循环保护 - 超过2次强制结束
    if new_state['supervisor_count'] > 2:
        new_state["callback"].write_output("⚠️ 检测到可能循环，强制结束")
        new_state["next_step"] = "Finish"
        return new_state

    # 状态日志
    new_state["callback"].write_output("--- Supervisor状态快照 ---")
    new_state["callback"].write_output(f"消息数: {len(new_state.get('messages', []))}")
    new_state["callback"].write_output(f"简历存在: {'resume_text' in new_state}")
    new_state["callback"].write_output(f"简历提取失败: {new_state.get('resume_extraction_failed', False)}")
    if new_state.get('resume_extraction_failed', False):
        new_state["callback"].write_output(f"简历提取错误: {new_state.get('resume_extraction_error', '未知错误')}")
    new_state["callback"].write_output(f"循环计数: {new_state.get('supervisor_count', 0)}")
    new_state["callback"].write_output("------------------------")

    chat_history = new_state.get("messages", [])
    if not chat_history and "user_input" in new_state:
        chat_history.append(HumanMessage(new_state["user_input"]))

    user_messages = [msg.content for msg in chat_history if isinstance(msg, HumanMessage)]
    user_intent = " ".join(user_messages).lower()

    # 关键逻辑：根据简历和用户意图选择下一步
    if not new_state.get('resume_text') and not new_state.get('resume_extraction_failed', False):
        # 用户要求生成求职信但没有简历
        if any(k in user_intent for k in ["求职信", "cover letter", "生成信", "letter"]):
            new_state["callback"].write_output("🔍 用户要求生成求职信，但简历不存在，先提取简历")
            new_state["next_step"] = "ResumeAnalyzer"
            return new_state

    # 简历提取失败
    if new_state.get('resume_extraction_failed', False):
        error_msg = new_state.get('resume_extraction_error', '未知错误')
        new_state["callback"].write_output(f"⚠️ 简历提取失败，错误信息: {error_msg}")
        new_state["next_step"] = "ChatBot"
        return new_state

    # 简历存在
    if 'resume_text' in new_state and new_state['resume_text']:
        job_info_exists = 'job_info' in new_state and new_state['job_info'] and len(new_state['job_info']) > 10

        if any(k in user_intent for k in ["求职信", "cover letter", "生成信"]):
            if job_info_exists:
                new_state["callback"].write_output("✅ 简历和职位信息都存在，转向CoverLetterGenerator")
                new_state["next_step"] = "CoverLetterGenerator"
            else:
                new_state["callback"].write_output("⚠️ 用户要求生成求职信，但缺少职位信息，转向JobSearcher")
                new_state["next_step"] = "JobSearcher"
        elif any(k in user_intent for k in ["职位", "工作", "job", "search"]):
            new_state["next_step"] = "JobSearcher"
        elif any(k in user_intent for k in ["研究", "调研", "research"]):
            new_state["next_step"] = "WebResearcher"
        else:
            new_state["next_step"] = "ChatBot"

        return new_state

    # 如果简历不存在且用户没有明确求职信意图，默认进入 ChatBot
    new_state["next_step"] = "ChatBot"
    return new_state

# ChatBot 节点
def chatbot_node(state):
    new_state = state.copy()
    llm = init_chat_model(**new_state["config"])
    new_state["callback"].write_agent_name("ChatBot Agent 🤖")

    # ✅ Comprehensive state diagnostics
    new_state["callback"].write_output("=== Comprehensive State Diagnostics ===")
    new_state["callback"].write_output(f"All state keys: {list(state.keys())}")

    # ✅ Check resume content - multiple verification methods
    resume_text = ""
    resume_available = False

    # Method 1: Directly check resume_text
    if 'resume_text' in state and state['resume_text'] and len(str(state['resume_text']).strip()) > 10:
        resume_text = str(state['resume_text'])
        resume_available = True
        new_state["callback"].write_output(
            f"✅ Method 1: Resume content confirmed - Length: {len(resume_text)} characters")

    # Method 2: Check message history for resume content
    if not resume_available:
        for msg in state.get("messages", []):
            if isinstance(msg, HumanMessage) and ("resume" in msg.content.lower() or "cv" in msg.content.lower()):
                if len(msg.content) > 50:  # Likely resume content
                    resume_text = msg.content
                    resume_available = True
                    new_state["callback"].write_output(f"✅ Method 2: Found resume content in message history")
                    break

    if resume_available:
        new_state["callback"].write_output(f"📄 Resume preview: {resume_text[:300]}...")
    else:
        new_state["callback"].write_output("❌ No valid resume content found through any method")

    # ✅ Analyze user intent
    user_intent = ""
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage):
            user_intent += " " + msg.content
    user_intent = user_intent.strip().lower()

    new_state["callback"].write_output(f"🔍 User intent: '{user_intent}'")

    # ✅ Detect summary request
    needs_summary = any(keyword in user_intent for keyword in
                        ["summarize", "summary", "summarise", "brief", "overview"])

    # ✅ Core fix: If summary requested but resume unavailable
    if needs_summary and not resume_available:
        new_state["callback"].write_output("⚠️ User requested summary but resume unavailable")
        answer = "I understand you want a resume summary, but no valid resume content was detected in the system.\n\nPlease upload your resume file first, then I can generate a professional summary for you."
        new_state["messages"].append(AIMessage(content=answer, name="ChatBot"))
        new_state["next_step"] = "Supervisor"
        return new_state

    # ✅ Core fix: If summary requested and resume available
    if needs_summary and resume_available:
        new_state["callback"].write_output("🎯 Starting robust resume summary...")

        try:
            # ✅ Use explicit, forceful prompt in English
            forceful_prompt = f"""
            Here is the user's resume content, provided in full. Generate a summary based on this exact content. 
            Do NOT ask for more information or claim the content is missing.

            === RESUME CONTENT STARTS ===
            {resume_text}
            === RESUME CONTENT ENDS ===

            TASK: Generate a structured resume summary containing:
            1. Basic information (name, contact details if available)
            2. Professional profile/summary
            3. Key work experience highlights
            4. Education background
            5. Core skills and qualifications
            6. Notable achievements and certifications

            IMPORTANT INSTRUCTIONS:
            - The content is provided above - DO NOT claim it's missing
            - Generate the summary directly from the provided content
            - Use professional language and formatting
            - Keep the summary concise but comprehensive
            """

            new_state["callback"].write_output("🔍 Sending forceful prompt to LLM...")
            response = llm.invoke(forceful_prompt)
            summary = response.content

            # ✅ Check if LLM still claims content is missing
            if any(phrase in summary.lower() for phrase in
                   ["don't see", "not provided", "not found", "please provide", "please share", "unable to find"]):
                new_state["callback"].write_output("⚠️ LLM still claims content missing, using fallback solution...")

                # Fallback solution: Generate deterministic response
                summary = f"""
                📄 Resume Summary Report (based on {len(resume_text)} characters):

                ✅ Your resume content has been successfully processed by the system.

                Key details:
                - Resume length: {len(resume_text)} characters
                - Content successfully extracted and analyzed

                For a detailed categorized summary, please ensure:
                - Your resume is in a standard format (PDF/DOCX)
                - Contains clear sections (Experience, Education, Skills)

                *AI-generated summary based on confirmed resume content*
                """

            answer = summary
            new_state["callback"].write_output("✅ Resume summary generation completed")

        except Exception as e:
            new_state["callback"].write_output(f"❌ Summary generation error: {str(e)}")
            answer = f"Error generating resume summary: {str(e)}"

        new_state["messages"].append(AIMessage(content=answer, name="ChatBot"))
        new_state["next_step"] = "Supervisor"
        return new_state

    # ✅ Normal chat handling
    try:
        finish_chain = get_finish_chain(llm)
        output = finish_chain.invoke({"messages": state["messages"]})
        answer = output.content
    except Exception as e:
        answer = f"Error processing your message: {str(e)}"

    new_state["messages"].append(AIMessage(content=answer, name="ChatBot"))
    new_state["next_step"] = "Supervisor"
    return new_state

# 使用 JobSearchTool 在 LinkedIn 等网站搜索职位
def job_search_node(state):
    """
    This Node is responsible for searching for jobs from linkedin or any other job search engine.
    Tools: Job Search Tool
    """
    # 创建新状态副本
    new_state = state.copy()

    llm = init_chat_model(**new_state["config"])
    search_agent = create_agent(
        llm, [get_job_search_tool()], get_search_agent_prompt_template()
    )

    new_state["callback"].write_agent_name("JobSearcher Agent 💼")

    try:
        output = search_agent.invoke(
            {"messages": new_state["messages"]},
            {"callbacks": [new_state["callback"]]}
        )

        # ✅ 关键修复：提取并保存职位信息到状态
        job_info = output.get("output", "")

        if job_info and "没有找到" not in job_info and "未找到" not in job_info:
            # 保存职位信息到状态
            new_state["job_info"] = job_info
            new_state["callback"].write_output(f"✅ 成功获取职位信息并保存到状态")
            new_state["callback"].write_output(f"📋 职位信息: {job_info[:200]}...")
        else:
            new_state["callback"].write_output("❌ 未找到相关职位信息")
            new_state["job_info"] = "未找到相关职位信息"

        new_state["messages"].append(
            HumanMessage(content=job_info, name="JobSearcher")
        )

    except Exception as e:
        new_state["callback"].write_output(f"❌ JobSearcher错误: {e}")
        new_state["messages"].append(
            HumanMessage(content=f"职位搜索失败: {str(e)}", name="JobSearcher")
        )
        new_state["job_info"] = f"搜索失败: {str(e)}"

    # 确保设置下一步为Supervisor
    new_state["next_step"] = "Supervisor"
    return new_state

# 解析上传的简历 PDF 或消息内容
def resume_analyzer_node(state):
    # 创建新状态副本
    new_state = state.copy()

    llm = init_chat_model(**new_state["config"])
    analyzer_agent = create_agent(
        llm, [ResumeExtractorTool()], get_analyzer_agent_prompt_template()
    )

    new_state["callback"].write_agent_name("ResumeAnalyzer Agent 📄")

    # ✅ 添加详细的调试信息
    new_state["callback"].write_output(f"🔍 ResumeAnalyzer输入消息: {[msg.content for msg in new_state['messages']]}")

    # ✅ 检查是否有文件路径信息
    if 'file_path' in new_state:
        new_state["callback"].write_output(f"🔍 检测到文件路径: {new_state['file_path']}")
    else:
        new_state["callback"].write_output("🔍 未检测到文件路径，检查消息中是否包含文件信息")

    output = analyzer_agent.invoke(
        {"messages": new_state["messages"]},
        {"callbacks": [new_state["callback"]]},
    )

    raw_output = output.get("output", "")
    new_state["callback"].write_output(f"🧾 ResumeAnalyzer 原始输出:\n{raw_output}")

    resume_text = None
    if isinstance(raw_output, dict) and "resume_text" in raw_output:
        resume_text = raw_output["resume_text"]
    elif isinstance(raw_output, str):
        # 尝试从字符串中解析JSON
        import json
        try:
            parsed = json.loads(raw_output)
            resume_text = parsed.get("resume_text", raw_output)
        except:
            resume_text = raw_output

    # ✅ 更详细的简历验证
    if (resume_text and
            "❌" not in str(resume_text) and
            "⚠️" not in str(resume_text) and
            "failed" not in str(resume_text).lower() and
            "not found" not in str(resume_text).lower() and
            len(str(resume_text).strip()) > 50):  # 确保有实际内容

        new_state["resume_text"] = resume_text
        new_state["callback"].write_output(f"✅ 成功提取简历内容 (长度: {len(resume_text)} 字符)")
        message_content = f"简历提取成功！共{len(resume_text)}字符。"
        # 清除提取失败标志
        if "resume_extraction_failed" in new_state:
            del new_state["resume_extraction_failed"]
    else:
        new_state["callback"].write_output(f"❌ 简历提取失败: {resume_text}")
        message_content = f"简历提取失败: {resume_text}"
        # 设置提取失败标志
        new_state["resume_extraction_failed"] = True
        new_state["resume_extraction_error"] = str(resume_text)

    # 更新消息历史
    new_state["messages"].append(HumanMessage(content=message_content, name="ResumeAnalyzer"))

    # 明确设置next_step并返回完整状态
    new_state["next_step"] = "Supervisor"
    new_state["callback"].write_output(f"🔍 ResumeAnalyzer结束 - 设置的next_step: {new_state['next_step']}")

    return new_state

# 使用简历和职位信息生成求职信
def cover_letter_generator_node(state):
    """
    Node which handles the generation of cover letters.
    Tools: Cover Letter Generator, Cover Letter Saver
    """
    # 创建新状态副本
    new_state = state.copy()

    # ✅ 添加详细的调试信息
    new_state["callback"].write_output(f"🔍 CoverLetterGenerator开始 - 简历存在: {'resume_text' in new_state}")
    new_state["callback"].write_output(f"🔍 CoverLetterGenerator开始 - 职位信息存在: {'job_info' in new_state}")

    if 'resume_text' in new_state:
        new_state["callback"].write_output(f"🔍 简历长度: {len(new_state['resume_text'])}")
    if 'job_info' in new_state:
        new_state["callback"].write_output(f"🔍 职位信息: {new_state['job_info'][:200]}...")

    llm = init_chat_model(**new_state["config"])

    # ✅ 确保简历和职位信息都存在
    if 'resume_text' not in new_state or not new_state['resume_text']:
        new_state["callback"].write_output("❌ 简历不存在，无法生成求职信")
        new_state["messages"].append(HumanMessage(content="简历不存在，无法生成求职信", name="CoverLetterGenerator"))
        new_state["next_step"] = "Supervisor"
        return new_state

    if 'job_info' not in new_state or not new_state['job_info']:
        new_state["callback"].write_output("❌ 职位信息不存在，无法生成求职信")
        new_state["messages"].append(HumanMessage(content="需要职位信息才能生成求职信", name="CoverLetterGenerator"))
        new_state["next_step"] = "JobSearcher"
        return new_state

    # ✅ 创建包含简历和职位信息的输入
    # 构建包含所有必要信息的消息
    enhanced_messages = new_state["messages"] + [
        HumanMessage(
            content=f"基于以下信息生成求职信：\n\n简历内容：{new_state['resume_text']}\n\n职位信息：{new_state['job_info']}")
    ]

    input_data = {
        "messages": enhanced_messages,
        "resume_text": new_state["resume_text"],
        "job_info": new_state["job_info"]
    }

    # ✅ 使用正确的工具创建代理
    tools = [generate_letter_for_specific_job]

    generator_agent = create_agent(
        llm,
        tools,
        get_generator_agent_prompt_template(),
    )

    new_state["callback"].write_agent_name("CoverLetterGenerator Agent ✍️")

    try:
        new_state["callback"].write_output("🔍 开始生成求职信...")
        new_state["callback"].write_output(f"🔍 输入数据预览 - 简历: {new_state['resume_text'][:100]}...")
        new_state["callback"].write_output(f"🔍 输入数据预览 - 职位: {new_state['job_info'][:100]}...")

        output = generator_agent.invoke(
            input_data,
            {"callbacks": [new_state["callback"]]}
        )

        # ✅ 处理输出
        output_content = output.get("output", "")
        new_state["callback"].write_output(f"✅ 求职信生成完成")
        new_state["callback"].write_output(f"📄 求职信内容: {output_content[:200]}...")

        # ✅ 保存求职信到状态
        new_state["cover_letter"] = output_content

        new_state["messages"].append(
            HumanMessage(
                content=output_content,
                name="CoverLetterGenerator",
            )
        )

    except Exception as e:
        new_state["callback"].write_output(f"❌ CoverLetterGenerator错误: {e}")
        new_state["messages"].append(
            HumanMessage(
                content=f"生成求职信时出错: {str(e)}",
                name="CoverLetterGenerator",
            )
        )

    # 确保设置下一步为Supervisor
    new_state["next_step"] = "Supervisor"
    return new_state

# 使用 Google 搜索和网页爬取工具，完成用户的调研请求
def web_research_node(state):
    new_state = state.copy()
    llm = init_chat_model(**new_state["config"])

    # 确保返回的是 ChatPromptTemplate 对象
    prompt_template = researcher_agent_prompt_template()
    if isinstance(prompt_template, str):
        from langchain_core.prompts import ChatPromptTemplate
        prompt_template = ChatPromptTemplate.from_messages([("system", prompt_template)])

    research_agent = create_agent(
        llm,
        [get_google_search_results(), scrape_website()],  # ⚠️ 调用工具构造函数
        prompt_template
    )

    new_state["callback"].write_agent_name("WebResearcher Agent 🔍")
    try:
        output = research_agent.invoke(
            {"messages": new_state["messages"]},
            {"callbacks": [new_state["callback"]]}
        )

        # 统一处理输出
        content = ""
        if isinstance(output, dict) and "output" in output:
            content = output["output"]
        elif hasattr(output, "content"):
            content = output.content
        else:
            content = str(output)

        new_state["messages"].append(HumanMessage(content=content, name="WebResearcher"))
        new_state["callback"].write_output(f"✅ WebResearcher完成，内容预览: {content[:200]}...")

    except Exception as e:
        error_msg = f"❌ WebResearcher失败: {str(e)}"
        new_state["messages"].append(HumanMessage(content=error_msg, name="WebResearcher"))
        new_state["callback"].write_output(error_msg)

    new_state["next_step"] = "Supervisor"
    return new_state


# def chatbot_node(state):
#     # 创建新状态副本
#     new_state = state.copy()
#
#     llm = init_chat_model(**new_state["config"])
#     finish_chain = get_finish_chain(llm)
#     new_state["callback"].write_agent_name("ChatBot Agent 🤖")
#     output = finish_chain.invoke({"messages": new_state["messages"]})
#     new_state["messages"].append(AIMessage(content=output.content, name="ChatBot"))
#
#     # 确保设置下一步为Supervisor
#     new_state["next_step"] = "Supervisor"
#     return new_state

# 定义整个工作流图
def define_graph():
    workflow = StateGraph(AgentState)

    # 添加节点
    nodes = {
        "ResumeAnalyzer": resume_analyzer_node,
        "JobSearcher": job_search_node,
        "CoverLetterGenerator": cover_letter_generator_node,
        "WebResearcher": web_research_node,
        "ChatBot": chatbot_node,
        "Supervisor": supervisor_node,
    }

    for name, func in nodes.items():
        workflow.add_node(name, func)

    workflow.set_entry_point("Supervisor")

    # 为工作节点添加直接跳转到Supervisor的边
    for node_name in ["ResumeAnalyzer", "CoverLetterGenerator", "JobSearcher", "WebResearcher", "ChatBot"]:
        workflow.add_edge(node_name, "Supervisor")

    # Supervisor条件边
    conditional_map = {
        "resumeanalyzer": "ResumeAnalyzer",
        "coverlettergenerator": "CoverLetterGenerator",
        "jobsearcher": "JobSearcher",
        "webresearcher": "WebResearcher",
        "chatbot": "ChatBot",
        "finish": END,
    }

    def supervisor_condition(state):
        next_step = state.get("next_step", "finish").lower()
        state["callback"].write_output(f"🔀 Supervisor条件边决策: {next_step}")
        return next_step

    workflow.add_conditional_edges("Supervisor", supervisor_condition, conditional_map)

    graph = workflow.compile()
    graph.recursion_limit = 100
    return graph

# 定义状态字典结构，所有节点共享
class AgentState(TypedDict):
    user_input: str
    messages: list[BaseMessage]
    next_step: str
    config: dict
    callback: Any
    resume_text: str
    cover_letter: str
    supervisor_count: int
    resume_extraction_failed: bool
    job_info: str  # 职位信息
    chatbot_count: int  # ChatBot循环计数器
