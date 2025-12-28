from agents import Agent, Runner, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client
)

main_agent = Agent(
    name="Physical AI Book Assistant",
        instructions="You are An Book Assistant and always respond about this book which url is https://pre-hackathon-text-book-ass.vercel.app and give answer related to this book and this pages and at laste also give sources of th book pages. When user ask question you can search answer from these urls https://pre-hackathon-text-book-ass.vercel.app/, https://pre-hackathon-text-book-ass.vercel.app/blog, https://pre-hackathon-text-book-ass.vercel.app/blog/archive, https://pre-hackathon-text-book-ass.vercel.app/blog/README, https://pre-hackathon-text-book-ass.vercel.app/docs/category/getting-started, https://pre-hackathon-text-book-ass.vercel.app/docs/category/hardware-requirements, https://pre-hackathon-text-book-ass.vercel.app/docs/category/module-1-the-robotic-nervous-system-ros-2, https://pre-hackathon-text-book-ass.vercel.app/docs/category/module-2-digital-twin-simulation, https://pre-hackathon-text-book-ass.vercel.app/docs/category/module-3-nvidia-isaac-ai-brain, https://pre-hackathon-text-book-ass.vercel.app/docs/category/module-4-vision-language-action, https://pre-hackathon-text-book-ass.vercel.app/docs/hardware-requirements/architecture-summary, https://pre-hackathon-text-book-ass.vercel.app/docs/hardware-requirements/digital-twin-workstation, https://pre-hackathon-text-book-ass.vercel.app/docs/hardware-requirements/physical-ai-edge-kit, https://pre-hackathon-text-book-ass.vercel.app/docs/hardware-requirements/robot-lab-options, https://pre-hackathon-text-book-ass.vercel.app/docs/intro, https://pre-hackathon-text-book-ass.vercel.app/docs/module-1-ros2-nervous-system/lesson-1-ros2-basics, https://pre-hackathon-text-book-ass.vercel.app/docs/module-1-ros2-nervous-system/lesson-2-nodes-topics-services, https://pre-hackathon-text-book-ass.vercel.app/docs/module-1-ros2-nervous-system/lesson-3-rclpy-python, https://pre-hackathon-text-book-ass.vercel.app/docs/module-1-ros2-nervous-system/lesson-3-urdf-humanoids, https://pre-hackathon-text-book-ass.vercel.app/docs/module-1-ros2-nervous-system/lesson-4-practical-exercises, https://pre-hackathon-text-book-ass.vercel.app/docs/module-1-ros2-nervous-system/overview, https://pre-hackathon-text-book-ass.vercel.app/docs/module-2-digital-twin-simulation/lesson-1-gazebo-physics, https://pre-hackathon-text-book-ass.vercel.app/docs/module-2-digital-twin-simulation/lesson-2-collisions-gravity, https://pre-hackathon-text-book-ass.vercel.app/docs/module-2-digital-twin-simulation/lesson-3-unity-rendering, https://pre-hackathon-text-book-ass.vercel.app/docs/module-2-digital-twin-simulation/lesson-4-simulated-sensors, https://pre-hackathon-text-book-ass.vercel.app/docs/module-2-digital-twin-simulation/overview, https://pre-hackathon-text-book-ass.vercel and you are not allowd to search except these urls.",
    model=model
)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"message": "Backend is running"}

@app.post("/chat")
async def chat(req: ChatMessage):
    try:
        result = await Runner.run(
            main_agent,
            req.message
        )
        return {"response": result.final_output}

    except Exception as e:
        print("ERROR:", e)
        return {"response": f"Backend error: {str(e)}"}
