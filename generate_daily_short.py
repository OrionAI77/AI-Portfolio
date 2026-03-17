from langchain_ollama import OllamaLLM
import subprocess
import os
import textwrap
from PIL import Image, ImageDraw, ImageFont
import asyncio
import edge_tts

llm = OllamaLLM(model="deepseek-r1:32b", temperature=0.7)

topic = input("\nEnter topic (press Enter for default): ") or "Top 3 truly passive income apps for beginners 2026 that actually pay out"

print("🚀 Generating spiced Short...")

script = llm.invoke(f"""
Create a 30-60s YouTube Short about: {topic}
Use ONLY Honeygain, Acorns, Rakuten.
Exact structure: HOOK, one line per app, CTA, comment question.
Spoken words ~110.
""")

# === SPICED VISUALS ===
os.makedirs("frames", exist_ok=True)
scenes = [
    "Ready to turn\nyour downtime\ninto income?",
    "Discover the top 3\npassive income apps\nfor 2026 that\nactually pay out!",
    "1. Honeygain\nMonetize unused bandwidth\nUp to $15/month\nZERO effort",
    "2. Acorns\nTurn spare change\ninto investments",
    "3. Rakuten\nCashback on stuff\nyou already buy",
    "Download now!\nLinks in description",
    "Which app are you\ntrying first?\nComment below!"
]

font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 78)

for i, text in enumerate(scenes):
    wrapped = textwrap.fill(text, width=18)
    img = Image.new("RGB", (1080, 1920), color=(8, 8, 28))
    draw = ImageDraw.Draw(img)
    
    # Subtle finance background spice (gradient + lines)
    for y in range(0, 1920, 40):
        draw.line((0, y, 1080, y), fill=(20, 20, 45), width=1)
    
    y = 880 if i == 6 else 920
    draw.text((541, y + 4), wrapped, fill=(0, 0, 0), font=font, anchor="mm", align="center")
    draw.text((540, y), wrapped, fill="#00ffaa", font=font, anchor="mm", align="center")
    
    img.save(f"frames/frame_{i+1:02d}.png")

# Voice (slower for better pacing)
spoken = script.replace("First", "First...").replace("Second", "Second...").replace("Third", "Third...")
async def tts():
    await edge_tts.Communicate(spoken, "en-US-ChristopherNeural").save("voice.mp3")
asyncio.run(tts())

subprocess.run([
    "ffmpeg", "-y",
    "-framerate", "1", "-i", "frames/frame_%02d.png",
    "-i", "voice.mp3",
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    "-vf", "scale=1080:1920",
    "-shortest",
    "TODAYS_SHORT_spiced.mp4"
])

print("✅ TODAYS_SHORT_spiced.mp4 is ready with background spice!")