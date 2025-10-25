from ast import List  # 导入 AST 模块的 List（其实这里不需要，用 typing.List 就够了）
from typing import Literal, Optional, List, Union  # 导入类型提示工具
# Literal: 限定字段只能取固定值
# Optional: 字段可选（允许 None）
# List: 列表类型
# Union: 字段可以是多种类型之一（例如 int 或 str）
from pydantic import BaseModel, Field  # 导入 Pydantic 基类和字段描述工具

# 定义 Supervisor/Agent 流程中“下一步动作”的数据模型
class RouteSchema(BaseModel):
    next_action: Literal[
        "ResumeAnalyzer",       # 简历分析 Agent
        "CoverLetterGenerator", # 求职信生成 Agent
        "JobSearcher",          # 职位搜索 Agent
        "WebResearcher",        # 网络研究 Agent
        "ChatBot",              # 聊天机器人
        "Finish",               # 流程完成
    ] = Field(
        ...,                    # 必填字段
        title="Next",           # 字段标题，用于文档或 UI 显示
        description="Select the next role",  # 字段描述
    )

# 定义职位搜索输入参数的数据模型
class JobSearchInput(BaseModel):
    keywords: str = Field(
        description="Keywords describing the job role. (if the user is looking for a role in particular company then pass company with keywords)"
        # 必填，职位关键词。如果要找特定公司职位，可把公司名加入关键词
    )
    location_name: Optional[str] = Field(
        description='Name of the location to search within. Example: "Kyiv City, Ukraine".'
        # 可选，搜索的城市或地区，例如 "Kyiv City, Ukraine"
    )
    employment_type: Optional[
        List[
            Literal[
                "full-time",   # 全职
                "contract",    # 合同制
                "part-time",   # 兼职
                "temporary",   # 临时
                "internship",  # 实习
                "volunteer",   # 志愿者
                "other",       # 其他
            ]
        ]
    ] = Field(description="Specific type(s) of job to search for.")
    # 可选，职位类型列表，可多选

    limit: Optional[int] = Field(
        default=5,  # 默认最多返回 5 条职位信息
        description="Maximum number of jobs to retrieve."  # 最大职位数量
    )

    job_type: Optional[List[Literal["onsite", "remote", "hybrid"]]] = Field(
        description="Filter for remote jobs, onsite or hybrid"
        # 可选，职位工作方式，可选择 onsite（现场）、remote（远程）、hybrid（混合）
    )

    experience: Optional[
        List[
            Literal[
                "internship",         # 实习
                "entry-level",        # 入门级
                "associate",          # 初级/助理
                "mid-senior-level",   # 中高级
                "director",           # 总监
                "executive",          # 高管
            ]
        ]
    ] = Field(
        description='Filter by experience levels. Options are "internship", "entry level", "associate", "mid-senior level", "director", "executive". pass the exact arguments'
        # 可选，经验等级列表，指定职位的经验要求
    )

    listed_at: Optional[Union[int, str]] = Field(
        default=86400,  # 默认 86400 秒，即 24 小时内发布的职位
        description="Maximum number of seconds passed since job posting. 86400 will filter job postings posted in the last 24 hours."
        # 可选，限制职位发布时间（秒数或字符串）
    )

    distance: Optional[Union[int, str]] = Field(
        default=25,  # 默认搜索半径 25 英里
        description="Maximum distance from location in miles. If not specified or 0, the default value of 25 miles is applied."
        # 可选，限制搜索范围距离（英里），未指定或为 0 时使用默认值
    )
