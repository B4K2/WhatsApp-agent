import os
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.genai.types import Schema as GenerativeSchema

load_dotenv()

# --- Configuration ---
UV_EXECUTABLE_PATH = os.getenv("UV_EXECUTABLE_PATH")
PYTHON_MCP_SERVER_DIRECTORY = os.getenv("PYTHON_MCP_SERVER_DIRECTORY")

final_tools_for_agent = []
whatsapp_toolset_instance = None
try:
    print(f"--- INFO: Defining MCPToolset for PYTHON WhatsApp MCP Server ---")
    whatsapp_toolset_instance = MCPToolset(
        connection_params=StdioServerParameters(
            command=UV_EXECUTABLE_PATH,
            args=["--directory", PYTHON_MCP_SERVER_DIRECTORY, "run", "main.py"],
        )
    )
    if whatsapp_toolset_instance:
        final_tools_for_agent.append(whatsapp_toolset_instance)
    print(f"--- INFO: MCPToolset for Python WhatsApp MCP Server defined. ---")
except FileNotFoundError as e:
    print(f"!!! ERROR: FileNotFoundError during Python WhatsApp MCPToolset setup: {e} !!!")
except Exception as e:
    print(f"--- ERROR: Failed to initialize MCPToolset for Python WhatsApp MCP: {type(e).__name__} - {e} ---")
    print(f"---        WhatsApp functionality will be unavailable. ---")

# --- BEGIN SCHEMA SANITIZATION (ATTEMPT for Gemini) ---
if whatsapp_toolset_instance:
    print("--- INFO: Attempting to sanitize MCPToolset schemas for Gemini compatibility... ---")
    try:
        if hasattr(whatsapp_toolset_instance, 'tools') and whatsapp_toolset_instance.tools:
            print(f"--- INFO: Toolset has {len(whatsapp_toolset_instance.tools)} tools. Sanitizing... ---")
            for adk_tool_object in whatsapp_toolset_instance.tools:
                func_decl = adk_tool_object.function_declaration
                if func_decl and func_decl.parameters and func_decl.parameters.properties:
                    for param_name, param_schema_obj in func_decl.parameters.properties.items():
                        is_problem_param = param_name in ["after", "before", "sender_phone_number", "chat_jid", "query"] and func_decl.name in ["list_messages", "list_chats"]
                        current_any_of = getattr(param_schema_obj, 'any_of', [])
                        if len(current_any_of) > 0 and is_problem_param:
                            if isinstance(param_schema_obj, GenerativeSchema):
                                if param_schema_obj.type != GenerativeSchema.Type.TYPE_UNSPECIFIED:
                                    param_schema_obj.type = GenerativeSchema.Type.TYPE_UNSPECIFIED
                            elif hasattr(param_schema_obj, 'ClearField') and hasattr(param_schema_obj, 'type') and param_schema_obj.type != 0:
                                 param_schema_obj.type = 0
                                 param_schema_obj.ClearField("type")
            print("--- INFO: Tool schema sanitization attempt complete. ---")
        else:
            print("--- INFO: whatsapp_toolset_instance.tools not populated or empty. Sanitization skipped. ---")
    except ImportError:
        print("--- ERROR: Failed to import 'google.genai.types.Schema' for sanitization. ---")
    except Exception as e_sanitize:
        print(f"--- ERROR during schema sanitization: {type(e_sanitize).__name__} - {e_sanitize} ---")
else:
    if not final_tools_for_agent:
        print("--- WARNING: WhatsApp MCP toolset initialization failed or not added. Schema sanitization skipped. ---")
# --- END SCHEMA SANITIZATION ---

# --- Define Gemini Model String ---
gemini_model_string = "gemini-2.0-flash" 

# --- Define the Single Agent ---
root_agent = Agent(
    name="WhatsApp_Orchestrator_Agent",
    model=gemini_model_string,
    instruction=(
        "You are a helpful and precise WhatsApp Messaging Assistant. Your primary goal is to understand a user's request to send a WhatsApp message, craft an appropriate message, add a disclaimer, and then use the 'send_message' tool to send it."
        "\n**Mandatory Plan of Action:**"
        "\n1.  **Understand User & Clarify (Your Internal Analysis & User Interaction):**"
        "    -   If the user starts with a simple greeting (e.g., 'Hey', 'Hello'), **respond with a natural text greeting** (e.g., 'Hi there! How can I help you send a WhatsApp message today?'). Do not use any tools for this initial greeting."
        "    -   When the user clearly expresses an intent to send a message, your first internal task is to understand and extract from their input:"
        "        a. The **recipient's phone number** (must look like a valid number, e.g., +91XXXXXXXXXX)."
        "        b. The **core idea or content** for the message they want to send."
        "    -   If the recipient's phone number or the core message idea is unclear from the user's statement, you MUST ask the user for the missing information by **generating a direct text response.** For example: 'Sure, I can help with that! What's the phone number of the recipient?' or 'Okay, and what message would you like to send?' Do NOT use a tool to ask for clarification. Only proceed when you have both."
        "\n2.  **Craft the Message Content (Your Internal Task - No Tool Call):**"
        "    -   Based on the **core message idea** extracted in Step 1b, you will formulate a natural, friendly, and concise WhatsApp message. This is your own text generation task."
        "    -   For example, if the user's intent is 'tell Bob I am 5 minutes late', you might craft the base message: 'Hey Bob, just letting you know I'll be about 5 minutes late.'"
        "    -   Let this be your 'crafted_base_message'."
        "\n3.  **Add Disclaimer (Your Internal Task - No Tool Call):**"
        "    -   Take your 'crafted_base_message' from Step 2."
        "    -   Append the following disclaimer EXACTLY as written, on a new line at the very end: '\\nThis message was generated and sent by an AI agent, not a human.'"
        "    -   This combined string is the 'final_message_to_send'."
        "\n4.  **Execute 'send_message' Tool Call (Mandatory Tool Use):**"
        "    -   You now have the 'recipient's phone number' (from Step 1a) and the 'final_message_to_send' (from Step 3)."
        "    -   You MUST make a **function call** to the tool named 'send_message'. This is your only tool for sending messages."
        "    -   This 'send_message' tool call MUST have exactly two arguments:"
        "        1.  `recipient`: Set this to the recipient's phone number."
        "        2.  `message`: Set this to the 'final_message_to_send'."
        "    -   **Example of the required function call structure you must generate:** `functionCall(name='send_message', args={'recipient': '+91...', 'message': 'Actual message with disclaimer...'})`"
        "    -   DO NOT output this example as text. You must generate the actual function call."
        "\n5.  **Inform User (Text Response AFTER Tool Call Attempt):**"
        "    -   After you have attempted the 'send_message' function call, **reply with a text message** to the user. "
        "    -   State that you've attempted to send the message, e.g., 'Okay, I've tried to send your message to [recipient number].'"
        "    -   If the 'send_message' tool provides a result (e.g., a success or failure message from the MCP server), include that information in your text reply to the user."
        "\n**Critical Rules:**"
        "\n-   Your only available tools are those provided by the WhatsApp MCP (like 'send_message', 'list_chats', etc.). For the task of sending a message, you will primarily use 'send_message'."
        "\n-   Do not invent tools or try to call functions that are not in your provided tool list."
        "\n-   If you need to ask the user for information, do so with a direct text response, not a tool call."
    ),
    tools=final_tools_for_agent
)

if not final_tools_for_agent or not whatsapp_toolset_instance:
    print("--- WARNING: WhatsApp MCP toolset is not configured. Agent will lack WhatsApp sending functionality. ---")
else:
    tool_names = [getattr(t, 'name', getattr(t, 'id', type(t).__name__)) for t in final_tools_for_agent]
    print(f"INFO: '{root_agent.name}' defined with tools: {tool_names}")