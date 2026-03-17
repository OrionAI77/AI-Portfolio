import asyncio
import platform

async def main():
    responses = {
        "hi": "Hey there! I'm Orion's chatbot—how can I help?",
        "help": "Just say 'hi' to chat or 'bye' to exit!",
        "bye": "Catch you later—thanks for chatting!",
        "default": "Sorry, I didn’t get that. Try 'help' for options."
    }
    
    print("Chatbot started! Type 'bye' to exit.")
    while True:
        user_input = input("> ").lower()
        reply = responses.get(user_input, responses["default"])
        print(reply)
        if user_input == "bye":
            break
        await asyncio.sleep(0.1)  # Keep it browser-friendly with Pyodide

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())