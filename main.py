import requests
from agents import Agent, Runner, function_tool, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, set_tracing_disabled
import os
from dotenv import load_dotenv
from rich import print
import rich

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY IS NOT SET. PLEASE MAKE SURE IT IS.")


external_client = AsyncOpenAI(
    api_key = gemini_api_key,
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
)
                      

model = OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash",
    openai_client = external_client
)

config = RunConfig(
    model = model,
    model_provider = external_client,     
    tracing_disabled = True
)

@function_tool
def get_products():
    """
    fetches a list of products from template-03-api.vercel.app/api
    """
    url = "https://hackathon-apis.vercel.app/api/products"
    try:
        response = requests.get(url)
        data = response.json()
        products = []
        return [
            {
                
             "productName": p.get("productName"),    
             "category": p.get("category"),
             "price": p.get("price"),
             "inventory": p.get("inventory"),
             "colors": p.get("colors"),
             "IsNew": p.get("IsNew"),
             "status": p.get("status"),
             "image": p.get("image"),
             "description": p.get("description"),
            }
            for p in data
        ]
    except requests.RequestException as e:
        return {"error": str(e)}
 


Shopping_agent = Agent(
    name = "Shopping Agent",
    instructions= "You are a shopping agent. Assist users with shopping and its queries.",
    tools = [get_products],
    model = model
    
)

shopping_queries = [
    "Show me a short and clean list of products(just names and prices please).",
    "What's the price of all the products?",
    "Are there any discounts today?",
    "Do you have any buy-one-get-one-free offers?",
    "Is this product in stock?",
    "When will this item be restocked?",
    "How long will delivery take?",
    "Do you offer same-day delivery in Karachi?",
    "What are the shipping charges?",
    "Can I get free delivery on my order?",
    "How do I place an order?",
    "Can I pay with cash on delivery?",
    "Where can I track my order?",
    "Can I cancel my order after payment?",
    "What's your return policy?",
]
for query in shopping_queries:
    rich.print(f"\n[b purple]🙂 User prompt:[/b purple] {query}")
    result = Runner.run_sync(
    Shopping_agent, input = query, run_config = config
    )
    rich.print(f"[yellow] Agent Response [/yellow]{result.final_output}") 
