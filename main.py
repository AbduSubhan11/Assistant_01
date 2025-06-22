import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
from agents.run import RunConfig
from fastapi.responses import JSONResponse

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Environment setup
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Pydantic model
class PromptInput(BaseModel):
    prompt: str

# Define function tools
@function_tool
def get_contact_info():
    return (
        "You can contact Abdu Subhan via:\n"
        "- Email: abdusubhan6678@gmail.com\n"
        "- GitHub: https://github.com/abduSubhan11/\n"
        "- Or fill the contact form on his portfolio: https://the-subhan-portfolio.netlify.app/"
    )

@function_tool
def get_current_work():
    return (
        "Abdu Subhan is currently working at Xntric as a JAMstack Developer. "
        "He is also a freelance Web Developer, open to remote or contract-based opportunities."
    )

@function_tool
def get_location():
    return "Abdu Subhan is based in Pakistan, and is available for remote work worldwide."

@function_tool
def get_bio():
    return (
        "Abdu Subhan is a passionate Web Developer specializing in full-stack solutions. "
        "He builds modern, performant websites and apps using MERN Stack, JAMstack, GSAP, Framer Motion, and Three.js. "
        "Currently exploring Agentic AI technologies and building intelligent applications."
    )

@function_tool
def get_skill():
    return (
        "Abdu Subhan's technical skillset includes:\n"
        "- 🌐 Frontend: HTML, CSS, JavaScript, TypeScript, Tailwind CSS, React, Next.js\n"
        "- 🛠️ Backend: Node.js, Express.js, Python\n"
        "- 🗄️ Database: MongoDB\n"
        "- 🎨 Animation: GSAP, Framer Motion, Three.js\n"
        "- 🔧 Tools: Git, VS Code, Postman, Vercel, Netlify, Figma"
    )

@function_tool
def get_experience():
    return (
        "Professional experience:\n"
        "- Xntric: JAMstack Developer (2024–present)\n"
        "- Freelance: Full-stack Web Developer (2024–present)\n"
        "- Built several production-grade apps using modern stacks.\n"
        "Portfolio: https://the-subhan-portfolio.netlify.app/"
    )

@function_tool
def get_education():
    return (
        "Education:\n"
        "- Agentic AI Training at GIAIC (Governor Initiative for AI & Computing)\n"
        "- Full-stack Web Development via online certifications\n"
        "- Intermediate (Pre-Engineering), Pakistan"
    )

@function_tool
def get_projects():
    return (
        "Some projects by Abdu Subhan:\n"
        "- 🛒 E-Commerce Store (React + Node.js)\n"
        "- 📖 Fullstack Blog (Next.js + MongoDB)\n"
        "- 🌐 Animated Portfolio (GSAP, Framer Motion)\n"
        "- 🤖 AI Chatbot (OpenAI Agents SDK)\n"
        "- 🔗 View all: https://the-subhan-portfolio.netlify.app/ and GitHub: https://github.com/abduSubhan11/"
    )

# AI Setup
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant answering any questions about Abdu Subhan or subhan both are same.",
    tools=[
        get_contact_info,
        get_current_work,
        get_location,
        get_bio,
        get_skill,
        get_experience,
        get_education,
        get_projects,
    ]
)

@app.post("/ask")
async def ask_agent(data: PromptInput):
    try:
        result = await Runner.run(agent, data.prompt, run_config=config)
        return {"response": result.final_output}
    except Exception as e:
        print(f"🔥 Agent Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
