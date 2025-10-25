# define tools
import os
import asyncio
from dotenv import load_dotenv
from pydantic import Field
from langchain.tools import BaseTool, tool, StructuredTool
from data_loader import load_resume, write_cover_letter_to_doc
from schemas import JobSearchInput
from search import get_job_ids, fetch_all_jobs
from utils import FireCrawlClient, SerperClient

load_dotenv()


# 根据用户指定条件在 LinkedIn 搜索职位
def linkedin_job_search(
    keywords: str,
    location_name: str = None,
    job_type: str = None,
    limit: int = 5,
    employment_type: str = None,
    listed_at=None,
    experience=None,
    distance=None,
) -> dict:  # type: ignore
    """
    Search LinkedIn for job postings based on specified criteria. Returns detailed job listings.
    """
    job_ids = get_job_ids(
        keywords=keywords,
        location_name=location_name,
        employment_type=employment_type,
        limit=limit,
        job_type=job_type,
        listed_at=listed_at,
        experience=experience,
        distance=distance,
    )
    job_desc = asyncio.run(fetch_all_jobs(job_ids))
    return job_desc

# 将 LinkedIn 搜索封装为 StructuredTool
def get_job_search_tool():
    """
    Create a tool for the JobPipeline function.
    Returns:
    StructuredTool: A structured tool for the JobPipeline function.
    """
    job_pipeline_tool = StructuredTool.from_function(
        func=linkedin_job_search,
        name="JobSearchTool",
        description="Search LinkedIn for job postings based on specified criteria. Returns detailed job listings",
        args_schema=JobSearchInput,
    )
    return job_pipeline_tool

# 提取上传的简历 PDF 文本
class ResumeExtractorTool(BaseTool):
    """
    Extract the content of a resume from a PDF file.
    Returns:
        dict: The extracted content of the resume.
    """
    name: str = "ResumeExtractor"
    description: str = "Extract the content of uploaded resume from a PDF file."

    def extract_resume(self) -> str:
        """
        Extract resume content from a PDF file.
        """
        import fitz  # PyMuPDF

        temp_path = os.path.join("temp", "resume.pdf")

        if not os.path.exists(temp_path):
            return "❌ No resume file found in temp directory. Please upload again."

        text = ""
        with fitz.open(temp_path) as pdf:
            for page in pdf:
                text += page.get_text("text")

        if not text.strip():
            return "⚠️ Resume PDF is empty or unreadable."

        return text

    def _run(self) -> dict:
        return {"resume_text": self.extract_resume()}


# Cover Letter Generation Tool
@tool
def generate_letter_for_specific_job(resume_details: str, job_details: str) -> dict:
    """
    Generate a tailored cover letter using the provided CV and job details. This function constructs the letter as plain text.
    returns: A dictionary containing the job and resume details for generating the cover letter.
    """
    return {"job_details": job_details, "resume_details": resume_details}

# 将生成的求职信保存为 Word 文档，并返回下载路径。
@tool
def save_cover_letter_for_specific_job(
    cover_letter_content: str, company_name: str
) -> str:
    """
    Returns a download link for the generated cover letter.
    Params:
    cover_letter_content: The combine information of resume and job details to tailor the cover letter.
    """
    filename = f"temp/{company_name}_cover_letter.docx"
    file = write_cover_letter_to_doc(cover_letter_content, filename)
    abs_path = os.path.abspath(file)
    return f"Here is the download link: {abs_path}"


# Web 搜索工具
@tool("google_search")
def get_google_search_results(
    query: str = Field(..., description="Search query for web")
) -> str:
    """
    search the web for the given query and return the search results.
    """
    response = SerperClient().search(query)
    items = response.get("items")
    string = []
    for result in items:
        try:
            string.append(
                "\n".join(
                    [
                        f"Title: {result['title']}",
                        f"Link: {result['link']}",
                        f"Snippet: {result['snippet']}",
                        "---",
                    ]
                )
            )
        except KeyError:
            continue

    content = "\n".join(string)
    return content

# 网站爬取工具
@tool("scrape_website")
def scrape_website(url: str = Field(..., description="Url to be scraped")) -> str:
    """
    Scrape the content of a website and return the text.
    """
    try:
        content = FireCrawlClient().scrape(url)
    except Exception as exc:
        return f"Failed to scrape {url}"
    return content
