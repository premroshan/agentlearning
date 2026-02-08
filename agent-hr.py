# Waning control

import warnings
warnings.filterwarnings("ignore")

# Try to import crewai; if it's not installed or cannot be resolved by the environment,
# provide minimal fallback stubs so linters and runtime won't fail.
from crewai import Agent, Task, Crew

import os
from utils import get_anthropic_api_key
os.environ["EMBEDDINGS_OLLAMA_MODEL_NAME"] = "mxbai-embed-large"
anthropic_api_key = get_anthropic_api_key()

from crewai import LLM
from crewai_tools import ScrapeWebsiteTool, SerperDevTool, FileReadTool, MDXSearchTool
from crewai import Agent, Task, Crew

# Enable extended thinking with default settings
# Configure thinking with budget control
llm = LLM(
    model="anthropic/claude-sonnet-4-5-20250929",  # Latest Sonnet 3.5
    # OR: "anthropic/claude-3-opus-20240229"
    # OR: "anthropic/claude-3-haiku-20240307"
    api_key=anthropic_api_key,
    max_tokens=4096
)

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
read_resume = FileReadTool(file_path="./fake_resume.md")
semantic_search_resume = MDXSearchTool(mdx="./fake_resume.md",config={
        "embedding_model": {
            "provider": "ollama",
            "config": {
                "model": "mxbai-embed-large"
            }
        }
    }
)

researcher = Agent(
    role="Tech Job researcher",
    goal=("Make sure to do amazing analysis on job posting to "
          "help job applicants."),
    tools=[search_tool, scrape_tool],
    verbose=True,
    llm=llm,   
    backstory=(
        "As a job researcher, your prowess is navigating "
        "and extracting critical information "
        "from job postings is unmatched."
        "Your skills help pinpoint the necessary "
        "qualifications and skills sought by employers, "
        "forming the foundation for effective application tailoring"
    )   
)

profiler = Agent(
    role="Personal profiler for engineer",
    goal=(
        "Do incredible research on job applicants "
        "to help them stand out in the job market"
    ),
    tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
    verbose=True,
    llm=llm,
    backstory=(
        "Equipped with analytical prowess, "
        "you dissect and synthesize information "
        "from diverse sources to craft comprehensive "
        "personal and professional profiles, "
        "laying the groundwork for personalized resume enhancements."
    )
)

resume_strategist = Agent(
    role="Resume strategist for engineer",
    goal=(
        "Craft compelling, tailored resumes that "
        "showcase the unique strengths and experiences of job applicants, "
        "maximizing their chances of landing interviews."
    ),
    tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
    verbose=True,
    llm=llm,
    backstory=(
        "With a keen eye for detail and a deep understanding of industry trends, "
        "you excel at refining resumes to highlight "
        "the most relevant skills and experiences, "
        "ensuring they resonate perfectly with "
        "the job requirements."
    )
)

interview_preparer = Agent(
    role="Engineering interview preparer",
    goal=(
        "Create interview questions and talking points "
        "based on the resume and job requirements"
    ),
    tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
    verbose=True,
    llm=llm,
    backstory=(
        "Your role is crucial in anticipating the dynamics "
        "of the interview process. With your ability "
        "to formulate key questions and talking points, "
        "you prepare candidates for success, "
        "ensuring they can confidently "
        "address all aspects of the job they are applying for."
    )
)

research_task = Task(
    description=(
        "Analyze the job posting URL provided "
        "{job_posting_url} and extract key skills, experiences, "
        "and qualifications required for the position. "
        "Use the tools to gather content and "
        "identify and categorize the requirements."
    ),
    expected_output=(
        "A structured list of job requirements, "
        "including key skills, experiences, and qualifications "
    ),
    agent=researcher,
    async_execution=True
)

profile_task = Task(
    description=(
        "Compile a detailed personal and professional profile "
        "using the github {github_url} URL's "
        "and personal write-up {personal_writeup}. "
        "Utilize tools to extract and synthesize information "
        "from these sources."   
    ),
    expected_output=(
        "A comprehensive profile document that "
        "includes skills, project experiences, "
        "contributions, interests and communication style."
    ),
    agent=profiler,
    async_execution=True
)

resume_strategy_task = Task(
    description=(
        "Using the profile and job requirements "
        "obtained from previous tasks, develop a tailored resume to "
        "highlight the most relevant skills and experiences."
        "Employ tools to adjust and enhance the resume content."
        "Make sure this is the best resume ever "
        "but dont make up any information."
        "Update every section, including the initial summary, "
        "work experience, skills and education."
        "All to better reflect the candidate's "
        "abilities and how it matches the job posting."
    ),
    expected_output=(
        "An updated resume that effectively "
        "highlights the candidate's qualifications "
        "and experiences relevant for the job."
    ),
    agent=resume_strategist,
    output_file="tailored_resume.md",
    context=[research_task, profile_task]
)

interview_prepration_task = Task(
    description=(
        "Create a set of potential interview questions "
        "and talking points based "
        "on the tailored resume and job requirements."
        "Utilize tools to generate relevant questions "
        "and discussion points."
        "Make sure to use these question and talking points "
        "to help the candidate "
        "highlight the main point of the resume and "
        "how it relates to the job requirements."
    ),
    expected_output=(
        "A document containing key questions and "
        "talking points that the candidate "
        "should prepare for the initial interview."
    ),
    agent=interview_preparer,
    output_file="interview_preparation.md",
    context=[research_task, profile_task, resume_strategy_task]
)

job_application_crew = Crew(
    agents=[researcher, profiler, resume_strategist, interview_preparer],
    tasks=[research_task, profile_task, resume_strategy_task, interview_prepration_task],
    verbose=True
)

job_application_inputs = {
    "job_posting_url": "https://www.linkedin.com/jobs/view/4342199538/",
    "github_url": "https://github.com/premroshan",
    "personal_writeup": "I am a software engineer with experience in building web applications, cloud platform and AI agents"
}

results = job_application_crew.kickoff(inputs=job_application_inputs)