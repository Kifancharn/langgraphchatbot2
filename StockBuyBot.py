from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
import os

# Set API key directly (since .env not working)
os.environ["GOOGLE_API_KEY"] = "ADD API KEY"

# Define available stocks
AVAILABLE_STOCKS = {
    "MSFT": 200.3,
    "AAPL": 100.4,
    "AMZN": 150.0,
    "RIL": 87.6
}

class State(TypedDict):
    messages: Annotated[list, add_messages]


# ------------------- TOOLS -------------------
@tool
def get_stock_price(symbol: str) -> float:
    """Return the current price of a stock given the stock symbol"""
    return AVAILABLE_STOCKS.get(symbol.upper(), 0.0)


@tool
def buy_stocks(symbol: str, quantity: int, total_price: float) -> str:
    """Buy stocks given the stock symbol and quantity"""
    decision = interrupt(f"Approve buying {quantity} {symbol} stocks for ${total_price:.2f}?")

    if decision.lower() == "yes":
        return f"You bought {quantity} shares of {symbol} for a total price of ${total_price:.2f}."
    else:
        return "Buying declined."


tools = [get_stock_price, buy_stocks]


# ------------------- LLM SETUP -------------------
llm = init_chat_model("google_genai:gemini-2.0-flash")
llm_with_tools = llm.bind_tools(tools)


def chatbot_node(state: State):
    msg = llm_with_tools.invoke(state["messages"])
    return {"messages": [msg]}


# ------------------- GRAPH SETUP -------------------
memory = MemorySaver()
builder = StateGraph(State)
builder.add_node("chatbot", chatbot_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")
builder.add_edge("chatbot", END)
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "buy_thread"}}


# ------------------- INTERACTIVE FLOW -------------------

# Step 1: Ask user to choose a stock
print("\n📈 Available stocks:")
for stock, price in AVAILABLE_STOCKS.items():
    print(f" - {stock} : ${price}")

selected_stock = input("\nPlease select a stock symbol (e.g., MSFT): ").upper()
if selected_stock not in AVAILABLE_STOCKS:
    print("❌ Invalid stock symbol. Exiting.")
    exit()

quantity = int(input(f"Enter quantity of {selected_stock} stocks to check price for: "))

# Step 2: Ask AI for current price
state = graph.invoke(
    {"messages": [{"role": "user", "content": f"What is the current price of {quantity} {selected_stock} stocks?"}]},
    config=config
)
print("\n🤖 AI Response:", state["messages"][-1].content)

# Step 3: Ask AI to buy the selected stock
state = graph.invoke(
    {"messages": [{"role": "user", "content": f"Buy {quantity} {selected_stock} stocks at current price."}]},
    config=config
)
print("\n🛑 AI needs confirmation:", state.get("__interrupt__"))

# Step 4: Ask for user confirmation
decision = input("Approve (yes/no): ")
state = graph.invoke(Command(resume=decision), config=config)
print("\n✅ Final AI Response:", state["messages"][-1].content)
