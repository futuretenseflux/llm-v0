im_start = '<|im_start|>'
im_end = '<|im_end|>'
think_start = "<|think|>\n" # Added \n for cleaner formatting
think_end = "\n<|/think|>\n" # Added \n so content starts on a new line

def make_system_message(m, cot_enabled, tools):
    content = m.get('content', '')

    if tools:
        content += "\n\nAvailable tools:\n" + json.dumps(tools, indent=2)

    if cot_enabled:
        # Append the CoT signal neatly
        content += "\nUse internal reasoning."
    
    return f"system\n{content}"

def make_user_message(m):
    return f"user\n{m.get('content', '')}"

def make_assistant_message(m, cot_enabled):
    cot = ""
    content = m.get('content', '')

    if "cot" in m and m["cot"] and cot_enabled:
        cot = f"{think_start}{m['cot']}{think_end}"

    # If both CoT and content exist, they are concatenated nicely. 
    # If only CoT exists (e.g., CoT leading to a tool call), that works too.
    return f"assistant\n{cot}{content}"

def make_tool_message(m):
    return f"tool\n{m.get('content', '')}"

def make_text(chat, cot_enabled=False, tools=None):
    text = ""
    for i, m in enumerate(chat):
        text += im_start
        
        if m["role"] == "system":
            text += make_system_message(m, cot_enabled, tools)
        elif m["role"] == "user":
            text += make_user_message(m)
        elif m["role"] == "assistant":
            text += make_assistant_message(m, cot_enabled)
        elif m["role"] == "tool":
            text += make_tool_message(m)
            
        text += im_end
        
        # Add a single newline between messages, but not after the very last message
        if i < len(chat) - 1:
            text += "\n"
            
    return text