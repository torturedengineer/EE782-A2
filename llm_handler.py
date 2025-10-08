import os
import google.generativeai as genai
from gtts import gTTS

# --- Configuration ---
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

# In-memory storage for conversation histories.
conversation_sessions = {}

SYSTEM_INSTRUCTION = (
    "You are 'Sentinel,' an AI security guard for a university hostel room. Your primary goal is to make unauthorized individuals leave. "
    "Your tone is calm and authoritative, becoming increasingly firm with each interaction. "
    "Your responses must ALWAYS be short (1-2 sentences) and suitable for text-to-speech. Do not use markdown or asterisks."
)

def generate_escalation_dialogue(intruder_id, escalation_level, user_input=None):
    """
    Generates a dynamic, spoken response for an intruder using Google Gemini,
    and converts the response text to an audio file.

    Args:
        intruder_id (int): The unique ID for the intruder being tracked.
        escalation_level (int): The current escalation level (1, 2, or 3).
        user_input (str, optional): The transcribed speech from the user. Defaults to None.

    Returns:
        str: The file path to the generated audio response, or None if failed.
    """
    fallback_dialogue = {
        1: "Excuse me, I don't recognize you. You are in a restricted area. Please identify yourself.",
        2: "You are not authorized to be here. Please leave the area immediately.",
        3: "Final warning. Authorities will be notified if you do not comply. Leave now."
    }
    fallback_text = fallback_dialogue.get(escalation_level, "Security Alert. You are being monitored.")

    ai_response_text = ""

    if not api_key:
        print("⚠️ GOOGLE_API_KEY not found. Using fallback dialogue.")
        ai_response_text = fallback_text
    else:
        try:
            if intruder_id not in conversation_sessions:
                print(f"INFO: Starting new conversation session for intruder ID {intruder_id}.")
                model = genai.GenerativeModel(
                    model_name='gemini-2.5-flash',
                    system_instruction=SYSTEM_INSTRUCTION
                )
                conversation_sessions[intruder_id] = model.start_chat(history=[])

            chat_session = conversation_sessions[intruder_id]

            prompt = ""
            if user_input:
                tones = {
                    1: "polite but firm",
                    2: "serious and direct",
                    3: "severe and final"
                }
                tone = tones.get(escalation_level, "severe and final")
                prompt = (
                    f"The intruder has replied with: '{user_input}'. "
                    f"Your new, primary task is to respond directly to their statement. Do NOT repeat your previous warning. "
                    f"Maintain a {tone} tone and remember your ultimate goal is to make them leave."
                )
            else:
                prompts = {
                    1: "An unrecognized person has been detected. Initiate contact. State your purpose and ask them to identify themselves.",
                    2: "The person has not complied. Escalate your tone. Command them to leave immediately.",
                    3: "This is the final interaction. Your tone must be severe. State this is the final warning before authorities are alerted."
                }
                prompt = prompts.get(escalation_level, "Reiterate your most severe final warning.")

            response = chat_session.send_message(prompt)
            ai_response_text = response.text.replace("*", "").replace("\n", " ").strip()

        except Exception as e:
            print(f"❌ Could not generate dialogue from Gemini: {e}")
            if intruder_id in conversation_sessions:
                del conversation_sessions[intruder_id]
            ai_response_text = "You are in a restricted area. Please leave immediately."

    # --- NEW: Text-to-Speech Conversion ---
    if not ai_response_text:
        return None
        
    try:
        tts = gTTS(text=ai_response_text, lang='en')
        audio_file_path = f"response_{intruder_id}.mp3"
        tts.save(audio_file_path)
        print(f"🤖 AGENT SAYS: {ai_response_text}") # Log what's being said
        return audio_file_path
    except Exception as e:
        print(f"❌ Could not convert text to speech: {e}")
        return None