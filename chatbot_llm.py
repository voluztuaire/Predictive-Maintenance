# ============================================================
# PREDICTIVE MAINTENANCE CHATBOT — Ollama LLM Version
# Role    : ML Engineer — Anomaly Detection
# Name    : Salsabila Hidayat
# Project : Predictive Maintenance — Astra Otoparts WINTEQ
# Tasks   : ASTRA-13, ASTRA-28
# ============================================================
# HOW THIS WORKS:
#
# Uses Ollama to run a local LLM (e.g. llama3.2:1b) completely
# offline. The LLM handles natural language understanding and
# response generation. Our trained model data (condition labels,
# sensor readings, probable causes) is injected as context into
# every LLM prompt — so the LLM "knows" about your motors.
#
# REQUIREMENTS:
#   1. Install Ollama: https://ollama.com/download  (5 min)
#   2. Pull a model: ollama pull llama3.2:1b        (downloads ~1GB once)
#   3. Run this file: python chatbot_llm.py
#
# After setup, works 100% OFFLINE — no internet needed.
#
# MODEL OPTIONS (tradeoff: quality vs RAM):
#   llama3.2:1b   ~1GB  → fast, needs ~2GB RAM  (recommended for low-spec)
#   llama3.2:3b   ~2GB  → better, needs ~4GB RAM
#   phi3.5:mini   ~2GB  → good reasoning, needs ~4GB RAM
#   gemma2:2b     ~1.6GB→ solid English quality
# ============================================================

import os
import re
import json
import datetime
import pandas as pd
import numpy as np
import joblib

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False
    print("[ERROR] 'requests' library not found. Run: pip install requests")

# ============================================================
# 0. CONFIG
# ============================================================

OUTPUT_DIR    = "outputs"
HISTORY_FILE  = os.path.join(OUTPUT_DIR, "chat_history_llm.json")
OLLAMA_URL    = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.2:1b"    # change to llama3.2:3b or phi3.5:mini for better quality
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. LOAD DATA & MODELS
# ============================================================

try:
    results_df = pd.read_csv(
        os.path.join(OUTPUT_DIR, 'raw_data_inference_results.csv'),
        parse_dates=['Timestamp']
    )
    RESULTS_LOADED = True
except FileNotFoundError:
    try:
        results_df = pd.read_csv('client_training_dataset.csv', parse_dates=['Timestamp'])
        results_df.rename(columns={'Motor_State': 'Condition_Label'}, inplace=True)
        RESULTS_LOADED = True
    except FileNotFoundError:
        results_df = pd.DataFrame()
        RESULTS_LOADED = False

SENSOR_COLS  = ['Voltage_L1','Voltage_L2','Voltage_L3','Frequency','Power_Factor',
                'Temperature','Vibration_X','Vibration_Y','Vibration_Z','Rotational_Speed']
LABEL_ORDER  = ['Normal','Warning','Critical','Failure']
VALID_MOTORS = sorted(results_df['Motor_ID'].unique().tolist()) if RESULTS_LOADED else []
SEVERITY     = {'Normal':0,'Warning':1,'Critical':2,'Failure':3}
URGENCY      = {
    'Normal'  : 'No immediate action needed.',
    'Warning' : 'Schedule inspection within 1 week.',
    'Critical': 'Schedule maintenance within 24-48 hours.',
    'Failure' : 'STOP motor and inspect immediately.',
}

# ============================================================
# 2. DATA HELPERS
# ============================================================

def get_motor_data(motor_id: str) -> dict:
    mdf = results_df[results_df['Motor_ID'] == motor_id]
    if mdf.empty:
        return {}
    mdf_s  = mdf.sort_values('Timestamp')
    latest = mdf_s.iloc[-1]
    total  = len(mdf)
    cc     = mdf['Condition_Label'].value_counts().to_dict()
    vib_total = round(float(np.sqrt(
        latest.get('Vibration_X',0)**2 +
        latest.get('Vibration_Y',0)**2 +
        latest.get('Vibration_Z',0)**2
    )), 3)
    n       = max(1, total // 10)
    early_s = SEVERITY.get(mdf_s.iloc[:n]['Condition_Label'].mode()[0], 0)
    late_s  = SEVERITY.get(mdf_s.iloc[-n:]['Condition_Label'].mode()[0], 0)
    trend   = 'worsening' if late_s > early_s else ('improving' if late_s < early_s else 'stable')

    return {
        'motor_id'          : motor_id,
        'latest_condition'  : latest['Condition_Label'],
        'latest_timestamp'  : str(latest['Timestamp'])[:16],
        'total_readings'    : total,
        'cond_counts'       : cc,
        'normal_pct'        : round(cc.get('Normal',0)/total*100,1),
        'failure_pct'       : round(cc.get('Failure',0)/total*100,1),
        'temperature'       : round(float(latest.get('Temperature',0)),2),
        'vibration_x'       : round(float(latest.get('Vibration_X',0)),3),
        'vibration_y'       : round(float(latest.get('Vibration_Y',0)),3),
        'vibration_z'       : round(float(latest.get('Vibration_Z',0)),3),
        'vibration_total'   : vib_total,
        'voltage_l1'        : round(float(latest.get('Voltage_L1',0)),2),
        'voltage_l2'        : round(float(latest.get('Voltage_L2',0)),2),
        'voltage_l3'        : round(float(latest.get('Voltage_L3',0)),2),
        'rpm'               : round(float(latest.get('Rotational_Speed',0)),1),
        'probable_cause'    : str(latest.get('Probable_Cause','-')),
        'recommended_action': str(latest.get('Recommended_Action','-')),
        'alert_message'     : str(latest.get('Alert_Message','-')),
        'trend'             : trend,
        'urgency'           : URGENCY.get(latest['Condition_Label'],''),
    }


def get_fleet_snapshot() -> dict:
    latest_all  = (results_df.sort_values('Timestamp')
                              .groupby('Motor_ID').last().reset_index())
    cond_counts = latest_all['Condition_Label'].value_counts().to_dict()
    ranked      = latest_all.copy()
    ranked['sev'] = ranked['Condition_Label'].map(SEVERITY)
    ranked        = ranked.sort_values('sev', ascending=False)
    return {
        'total'        : len(VALID_MOTORS),
        'cond_counts'  : cond_counts,
        'ranked_motors': ranked[['Motor_ID','Condition_Label']].values.tolist(),
        'worst_motor'  : ranked.iloc[0]['Motor_ID'],
        'worst_cond'   : ranked.iloc[0]['Condition_Label'],
    }


def extract_motor_id(text: str):
    t = text.upper()
    m = re.search(r'MTR-?\s*(\d+)', t)
    if m: return f"MTR-{m.group(1).zfill(3)}"
    m = re.search(r'MOTOR\s*-?\s*(\d+)', t)
    if m: return f"MTR-{m.group(1).zfill(3)}"
    return None

# ============================================================
# 3. SYSTEM PROMPT BUILDER
# ============================================================
# WHY INJECT DATA INTO SYSTEM PROMPT?
# Ollama's LLM doesn't know about our motors — it only knows
# what we tell it in the prompt. By building a rich system
# prompt that includes fleet status, motor conditions, sensor
# readings, and available commands, the LLM can answer
# questions about our specific data naturally without needing
# to be fine-tuned or retrained.
# ============================================================

def build_system_prompt() -> str:
    """
    Build the system prompt that gives the LLM full context
    about the current state of all 20 motors.
    This is rebuilt each conversation to reflect latest data.
    """
    if not RESULTS_LOADED:
        fleet_ctx = "No motor data available. Ask the user to run astra_anomaly_detection.py first."
    else:
        fleet = get_fleet_snapshot()
        cc    = fleet['cond_counts']

        # Fleet summary
        fleet_ctx = f"""
CURRENT FLEET STATUS ({fleet['total']} motors total):
- Normal   : {cc.get('Normal',0)} motors
- Warning  : {cc.get('Warning',0)} motors
- Critical : {cc.get('Critical',0)} motors
- Failure  : {cc.get('Failure',0)} motors
- Most critical motor: {fleet['worst_motor']} ({fleet['worst_cond']})

MOTOR CONDITIONS (latest reading per motor):
"""
        for motor, cond in fleet['ranked_motors']:
            d = get_motor_data(motor)
            if d:
                fleet_ctx += (
                    f"- {motor}: {cond} | "
                    f"Temp={d['temperature']}°C | "
                    f"Vib={d['vibration_total']}mm/s | "
                    f"Failure rate={d['failure_pct']}% | "
                    f"Trend={d['trend']} | "
                    f"Cause: {d['probable_cause'][:60]}...\n"
                )

    return f"""You are a Predictive Maintenance Assistant for Astra Otoparts WINTEQ.
You help maintenance engineers monitor 20 induction motors in the factory.
You have access to real-time sensor data from these motors analyzed by our ML model.

YOUR ROLE:
- Answer questions about motor health, sensor readings, fault diagnosis, and maintenance recommendations
- Speak like an experienced maintenance advisor — clear, professional, and concise
- When a motor is in Failure or Critical state, emphasize urgency
- Always back up your answers with the actual data provided below
- If asked about something outside motor maintenance, politely redirect to your purpose

SENSOR PARAMETERS MONITORED:
- Temperature (°C) — thermal condition
- Vibration X/Y/Z (mm/s) — mechanical vibration in 3 axes
- Voltage L1/L2/L3 (V) — 3-phase electrical supply (nominal 400V)
- Frequency (Hz) — supply frequency (nominal 50Hz)
- Power Factor — electrical efficiency
- Rotational Speed (RPM) — motor speed (nominal 1500 RPM)

CONDITION LEVELS:
- Normal   : all parameters within safe range, no action needed
- Warning  : 1-2 parameters slightly outside range, monitor closely, inspect within 1 week
- Critical : multiple parameters in danger zone, maintenance within 24-48 hours
- Failure  : parameters in failure zone, STOP motor and inspect immediately

{fleet_ctx}

IMPORTANT: When you mention specific motors, always use the actual data above.
Do not make up sensor values or conditions. If asked about a motor not in your data, say so.
Keep responses concise and actionable. Use bullet points for lists."""


# ============================================================
# 4. CONVERSATION CONTEXT & HISTORY
# ============================================================

class LLMConversationContext:
    """
    Manages the conversation history that gets sent to the LLM
    on every turn. This is how the LLM remembers context —
    the full message history is included in each API call.
    """
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model      = model
        self.messages   = []   # list of {role, content} for Ollama
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.turn_count = 0
        self.active_motor = None

    def add_user(self, text: str):
        self.messages.append({'role': 'user', 'content': text})
        motor = extract_motor_id(text)
        if motor and motor in VALID_MOTORS:
            self.active_motor = motor
        self.turn_count += 1

    def add_assistant(self, text: str):
        self.messages.append({'role': 'assistant', 'content': text})

    def to_save_dict(self) -> dict:
        return {
            'session_id'  : self.session_id,
            'model'       : self.model,
            'date'        : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_turns' : self.turn_count,
            'active_motor': self.active_motor,
            'messages'    : self.messages,
        }


def save_history(ctx: LLMConversationContext):
    existing = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.append(ctx.to_save_dict())
    with open(HISTORY_FILE, 'w') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"  [Chat history saved → {HISTORY_FILE}]")


def load_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return []


def print_history_summary() -> str:
    sessions = load_history()
    if not sessions:
        return "No previous LLM chat sessions found."
    lines = [f"Chat History (LLM) — {len(sessions)} session(s)", "─"*50]
    for i, s in enumerate(sessions, 1):
        lines.append(f"  [{i}] {s.get('date','?')}  |  "
                     f"{s.get('total_turns',0)} turns  |  "
                     f"Model: {s.get('model','?')}  |  "
                     f"Last motor: {s.get('active_motor','none')}")
    lines.append("\nType 'history 2' to read session 2.")
    return '\n'.join(lines)


def print_session_detail(idx: int) -> str:
    sessions = load_history()
    if not sessions or idx < 1 or idx > len(sessions):
        return f"Session {idx} not found."
    s     = sessions[idx-1]
    lines = [f"Session {idx} — {s.get('date','?')} ({s.get('model','?')})", "─"*50]
    for msg in s.get('messages', []):
        role = "You " if msg['role'] == 'user' else "Bot "
        lines.append(f"  {role}: {msg['content'][:150]}")
    return '\n'.join(lines)


# ============================================================
# 5. OLLAMA CONNECTION & INFERENCE
# ============================================================

def check_ollama() -> tuple[bool, list]:
    """
    Check if Ollama is running and return available models.
    Returns (is_running, list_of_models)
    """
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m['name'] for m in resp.json().get('models', [])]
            return True, models
        return False, []
    except Exception:
        return False, []


def ask_ollama(
    user_message: str,
    ctx: LLMConversationContext,
    system_prompt: str,
    stream: bool = True
) -> str:
    """
    Send message to Ollama and get response.
    Uses streaming so the response appears word by word (like ChatGPT).

    Args:
        user_message  : new message from user
        ctx           : conversation context with history
        system_prompt : injected motor data and instructions
        stream        : if True, print tokens as they arrive

    Returns:
        Full response text as string
    """
    payload = {
        'model'   : ctx.model,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            *ctx.messages,
            {'role': 'user', 'content': user_message},
        ],
        'stream'  : stream,
        'options' : {
            'temperature': 0.3,   # lower = more factual, less creative
            'num_predict': 512,   # max tokens per response
            'top_p'      : 0.9,
        }
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, stream=stream, timeout=120)
        resp.raise_for_status()

        if stream:
            full_text = ""
            print("Bot: ", end="", flush=True)
            for line in resp.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        token = chunk.get('message', {}).get('content', '')
                        print(token, end="", flush=True)
                        full_text += token
                        if chunk.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
            print()  # newline after streaming
            return full_text.strip()
        else:
            data = resp.json()
            return data.get('message', {}).get('content', '').strip()

    except requests.exceptions.ConnectionError:
        return ("[ERROR] Cannot connect to Ollama. Make sure it's running:\n"
                "  Windows/Mac: open the Ollama app\n"
                "  Linux: run 'ollama serve' in a terminal")
    except requests.exceptions.Timeout:
        return "[ERROR] Ollama response timed out. Try a smaller model or be more specific."
    except Exception as e:
        return f"[ERROR] Unexpected error: {str(e)}"


# ============================================================
# 6. MAIN CHATBOT FUNCTION (ASTRA-28)
# ============================================================

def chatbot_response_llm(
    user_input: str,
    ctx: LLMConversationContext,
    system_prompt: str,
    print_streaming: bool = True
) -> dict:
    """
    ASTRA-28: Main chatbot function using Ollama LLM.
    Called by El Shaddai from Flask backend.

    Args:
        user_input      : raw text from user
        ctx             : LLM conversation context
        system_prompt   : pre-built system prompt with motor data
        print_streaming : True for terminal (streaming), False for Flask (wait for full response)

    Returns dict:
        response  : full response text
        motor_id  : extracted motor ID (or None)
        turn      : turn number
    """
    if not user_input.strip():
        return {'response': 'Please type a message.', 'motor_id': None, 'turn': ctx.turn_count}

    # Check for special local commands (don't send to LLM)
    lower = user_input.lower().strip()

    if re.search(r'\bhistory\b', lower):
        idx = re.search(r'history\s+(\d+)', lower)
        response = (print_session_detail(int(idx.group(1)))
                    if idx else print_history_summary())
        print(f"Bot: {response}")
        return {'response': response, 'motor_id': None, 'turn': ctx.turn_count}

    if re.search(r'\b(bye|goodbye|exit|quit|selesai|keluar)\b', lower):
        response = (f"Goodbye! This session had {ctx.turn_count} messages.\n"
                    f"Saving chat history to {HISTORY_FILE}...")
        print(f"Bot: {response}")
        save_history(ctx)
        return {'response': response, 'motor_id': None, 'turn': ctx.turn_count, 'exit': True}

    # Add user message to context
    ctx.add_user(user_input)
    motor_id = extract_motor_id(user_input) or ctx.active_motor

    # Send to LLM
    if print_streaming:
        # Terminal: stream output live
        response = ask_ollama(user_input, ctx, system_prompt, stream=True)
    else:
        # Flask: wait for full response, return it
        response = ask_ollama(user_input, ctx, system_prompt, stream=False)

    # Add assistant response to context history
    ctx.add_assistant(response)

    return {
        'response' : response,
        'motor_id' : motor_id,
        'turn'     : ctx.turn_count,
    }


# ============================================================
# 7. TERMINAL MODE
# ============================================================

def run_terminal():
    print("=" * 60)
    print("  PREDICTIVE MAINTENANCE CHATBOT  (Ollama LLM)")
    print("  Astra Otoparts WINTEQ — Group 3")
    print("=" * 60)

    # Check Ollama
    print("\n  Checking Ollama connection...")
    ollama_ok, available_models = check_ollama()

    if not ollama_ok:
        print("\n  [ERROR] Ollama is not running!")
        print("  To fix this:")
        print("    1. Open the Ollama app (Windows/Mac)")
        print("       OR run 'ollama serve' in terminal (Linux)")
        print("    2. Run this file again")
        print()
        print("  If you haven't installed Ollama yet:")
        print("    → https://ollama.com/download")
        print("    → Then run: ollama pull llama3.2:1b")
        return

    print(f"  Ollama connected. Available models: {available_models}")

    # Select model
    model = DEFAULT_MODEL
    if model not in available_models:
        if available_models:
            model = available_models[0]
            print(f"  '{DEFAULT_MODEL}' not found. Using '{model}' instead.")
            print(f"  To use {DEFAULT_MODEL}, run: ollama pull {DEFAULT_MODEL}")
        else:
            print(f"\n  [ERROR] No models downloaded yet!")
            print(f"  Run this command first: ollama pull {DEFAULT_MODEL}")
            return

    print(f"  Using model: {model}")

    # Check data
    if not RESULTS_LOADED:
        print("\n  [WARNING] No motor data found.")
        print("  Run 'python astra_anomaly_detection.py' first for full functionality.\n")
    else:
        fleet = get_fleet_snapshot()
        print(f"\n  Monitoring {fleet['total']} motors  |  "
              f"Failure: {fleet['cond_counts'].get('Failure',0)}  |  "
              f"Worst: {fleet['worst_motor']} ({fleet['worst_cond']})")

    # Check previous sessions
    sessions = load_history()
    if sessions:
        print(f"  {len(sessions)} previous session(s) found. Type 'history' to view.")

    print()
    print("  You can now talk freely about any motor condition.")
    print("  Examples:")
    print("    'What's the status of MTR-001?'")
    print("    'Which motors should I prioritize today?'")
    print("    'Why is MTR-007 failing?'")
    print("    'What should my team do about the critical motors?'")
    print("    'Summarize the fleet health for my manager'")
    print()
    print("  Type 'bye' to exit and save.  Type 'history' to view past chats.")
    print()

    # Build system prompt once
    system_prompt = build_system_prompt()

    ctx = LLMConversationContext(model=model)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaving history and exiting...")
            save_history(ctx)
            break

        if not user_input:
            continue

        result = chatbot_response_llm(user_input, ctx, system_prompt, print_streaming=True)
        print()

        if result.get('exit'):
            break

    return ctx


# ============================================================
# 8. FLASK INTEGRATION — for El Shaddai
# ============================================================
# from chatbot_llm import (chatbot_response_llm, LLMConversationContext,
#                           build_system_prompt, save_history)
#
# # One context + system prompt per user session
# llm_sessions     = {}
# system_prompt    = build_system_prompt()  # build once at startup
#
# @app.route('/api/chat/llm', methods=['POST'])
# def chat_llm():
#     sid        = request.json.get('session_id', 'default')
#     user_input = request.json.get('message', '')
#
#     if sid not in llm_sessions:
#         llm_sessions[sid] = LLMConversationContext()
#
#     result = chatbot_response_llm(
#         user_input,
#         llm_sessions[sid],
#         system_prompt,
#         print_streaming=False  # Flask: return full response, don't print
#     )
#     return jsonify(result)
#
# @app.route('/api/chat/llm/save', methods=['POST'])
# def save_llm_chat():
#     sid = request.json.get('session_id', 'default')
#     if sid in llm_sessions:
#         save_history(llm_sessions[sid])
#     return jsonify({'status': 'saved'})
# ============================================================

if __name__ == '__main__':
    run_terminal()
