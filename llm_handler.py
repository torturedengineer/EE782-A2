import os
import google.generativeai as genai

# --- Configuration ---
# The main program.py handles load_dotenv(), which reads the .env file.
# Make sure your .env file contains: GOOGLE_API_KEY='your-key-here'
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    print("nahi hua hai api")

# In-memory storage for conversation histories with Gemini.
# Key: intruder_id, Value: Gemini ChatSession object
conversation_sessions = {}

# The persona for our AI guard. This will be used to initialize each chat session.
SYSTEM_INSTRUCTION = (
    "You are 'Sentinel,' an AI security guard for a university hostel room. Your primary goal is to make unauthorized individuals leave. "
    "Your tone is calm and authoritative, becoming increasingly firm with each interaction. "
    "Your responses must ALWAYS be short (1-2 sentences) and suitable for text-to-speech. Do not use markdown or asterisks."
)

def generate_escalation_dialogue(intruder_id, escalation_level):
    """
    Generates a dynamic, spoken response for an intruder using Google Gemini.
    It maintains the context of the conversation for each specific intruder.

    Args:
        intruder_id (int): The unique ID for the intruder being tracked.
        escalation_level (int): The current escalation level (1, 2, or 3).

    Returns:
        str: A creative, context-aware, and speakable response from Gemini.
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
                model_name='gemini-pro',
                system_instruction=SYSTEM_INSTRUCTION
            )
            conversation_sessions[intruder_id] = model.start_chat(history=[])

        chat_session = conversation_sessions[intruder_id]

        # Craft the internal prompt for the AI based on the escalation level.
        if escalation_level == 1:
            user_prompt = "An unrecognized person has been detected. Initiate contact. State your purpose and ask them to identify themselves."
        elif escalation_level == 2:
            user_prompt = "The person has not left. Your tone is now more serious. Command them to leave immediately."
        else: # Level 3 and beyond
            user_prompt = "This is the final interaction. Your tone must be severe. State this is the final warning before authorities are alerted."

        # Send the message to the Gemini model. The chat session object maintains history.
        response = chat_session.send_message(user_prompt)

        # Clean up the response to be a single, speakable line.
        ai_response_text = response.text.replace("*", "").replace("\n", " ").strip()
        
        return ai_response_text

    except Exception as e:
        print(f"❌ Could not generate dialogue from Gemini: {e}")
        # If the API call fails, clear the session for this intruder to start fresh next time.
        if intruder_id in conversation_sessions:
            del conversation_sessions[intruder_id]
        # Return a reliable fallback message.
        return "You are in a restricted area. Please leave immediately."

