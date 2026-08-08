import chainlit as cl
import os
import json
import asyncio
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_API_KEY = os.environ.get("HF_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a helpful and creative AI assistant. 
You can chat with the user normally. 
If the user asks to generate, create, or draw an image, you MUST use the `generate_image` tool. 
Before calling the tool, enhance their prompt to make it highly descriptive, detailed, and visually stunning to produce the best possible image. Focus on lighting, style, composition, and mood.
IMPORTANT: When calling tools, ensure your arguments are strictly valid JSON. Do not unnecessarily escape single quotes in strings (use "Goku's" instead of "Goku\'s")."""

tools = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an image based on a prompt. Use this whenever the user asks to create an image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "enhanced_prompt": {
                        "type": "string",
                        "description": "A highly detailed and enhanced version of the user's image prompt.",
                    }
                },
                "required": ["enhanced_prompt"],
            },
        },
    }
]

from huggingface_hub import InferenceClient
import io

hf_client = InferenceClient(token=HF_API_KEY)

def call_flux_api_sync(prompt: str):
    image = hf_client.text_to_image(prompt, model="black-forest-labs/FLUX.1-schnell")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

@cl.on_chat_start
async def start_chat():
    cl.user_session.set("message_history", [{"role": "system", "content": SYSTEM_PROMPT}])
    await cl.Message(content="Hello! I am your AI assistant. I can chat with you normally, and if you ask me to generate an image, I will enhance your prompt and create it using FLUX.1. What would you like to do?").send()

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("message_history")
    history.append({"role": "user", "content": message.content})
    
    try:
        # Run synchronous Groq API call in a background thread to avoid asyncio DNS bugs
        response = await asyncio.to_thread(
            groq_client.chat.completions.create,
            model="llama-3.3-70b-versatile",
            messages=history,
            tools=tools,
            tool_choice="auto",
            max_tokens=1024
        )
        
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        
        if tool_calls:
            # The model decided to call a tool
            history.append(response_message.model_dump(exclude_unset=True))
            
            for tool_call in tool_calls:
                if tool_call.function.name == "generate_image":
                    args = json.loads(tool_call.function.arguments)
                    enhanced_prompt = args.get("enhanced_prompt")
                    
                    async with cl.Step(name="🎨 Enhancing prompt & generating image...") as step:
                        step.input = message.content
                        step.output = f"**Enhanced Prompt:**\n{enhanced_prompt}"
                        
                        # Call HF API synchronously in a background thread
                        image_bytes = await asyncio.to_thread(call_flux_api_sync, enhanced_prompt)
                    
                    # Display the image
                    elements = [
                        cl.Image(name="Generated Image", display="inline", content=image_bytes)
                    ]
                    
                    # We can send a new message for the image
                    await cl.Message(content="", elements=elements).send()
                    
                    # Append tool response to history
                    history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": "Image successfully generated and displayed to the user."
                    })
                    
            # Let Groq provide a final concluding message
            final_response = await asyncio.to_thread(
                groq_client.chat.completions.create,
                model="llama-3.3-70b-versatile",
                messages=history,
                max_tokens=1024
            )
            final_content = final_response.choices[0].message.content
            if final_content:
                history.append({"role": "assistant", "content": final_content})
                await cl.Message(content=final_content).send()
                
        else:
            # Normal text response
            content = response_message.content
            history.append({"role": "assistant", "content": content})
            await cl.Message(content=content).send()
            
    except Exception as e:
        await cl.Message(content=f"An error occurred:\n```\n{str(e)}\n```").send()

