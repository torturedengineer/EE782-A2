import os
import google.generativeai as genai

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
    maintaining conversation context for each specific intruder.
    
    Args:
        intruder_id (int): The unique ID for the intruder being tracked.
        escalation_level (int): The current escalation level (1, 2, or 3).
        user_input (str, optional): The transcribed speech from the user. Defaults to None.
    """
    if not api_key:
        print("⚠️ GOOGLE_API_KEY not found. Using fallback dialogue.")
        fallback_dialogue = {
            1: "Excuse me, I don't recognize you. You are in a restricted area. Please identify yourself.",
            2: "You are not authorized to be here. Please leave the area immediately.",
            3: "Final warning. Authorities will be notified if you do not comply. Leave now."
        }
        return fallback_dialogue.get(escalation_level, "Security Alert. You are being monitored.")

    try:
        # Retrieve this intruder's chat session, or start a new one.
        if intruder_id not in conversation_sessions:
            print(f"INFO: Starting new conversation session for intruder ID {intruder_id}.")
            model = genai.GenerativeModel(
                model_name='gemini-2.5-flash',
                system_instruction=SYSTEM_INSTRUCTION
            )
            conversation_sessions[intruder_id] = model.start_chat(history=[])

        chat_session = conversation_sessions[intruder_id]

        # --- FIX: Overhauled prompt logic for true conversation ---
        prompt = ""
        if user_input:
            # If the user said something, the conversation is now primary.
            # The escalation level sets the tone, but we focus on their input.
            tones = {
                1: "polite but firm",
                2: "serious and direct",
                3: "severe and final"
            }
            tone = tones.get(escalation_level, "severe and final")
            prompt = (
                f"The person has responded with: '{user_input}'. "
                f"Address their response directly. Your tone must be {tone}. "
                "Keep your goal in mind: you need them to leave."
            )
        else:
            # If the user is silent, deliver the standard escalating warning.
            prompts = {
                1: "An unrecognized person has been detected. Initiate contact. State your purpose and ask them to identify themselves.",
                2: "The person has not left. Your tone is now more serious. Command them to leave immediately.",
                3: "This is the final interaction. Your tone must be severe. State this is the final warning before authorities are alerted."
            }
            prompt = prompts.get(escalation_level, "Reiterate your final warning. Be severe.")

        # Send the prompt to Gemini. The chat session maintains the conversation history.
        response = chat_session.send_message(prompt)

        ai_response_text = response.text.replace("*", "").replace("\n", " ").strip()
        
        return ai_response_text

    except Exception as e:
        print(f"❌ Could not generate dialogue from Gemini: {e}")
        if intruder_id in conversation_sessions:
            del conversation_sessions[intruder_id]
        return "You are in a restricted area. Please leave immediately."

