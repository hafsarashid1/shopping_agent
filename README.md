🛍️ AI Shopping Agent

Your personal AI-powered shopping assistant — built with "Google Gemini" and the "Agents framework".  
This project connects to a live product API, understands user shopping queries, and responds intelligently like a real virtual assistant 💬🤖  

---

🚀 Features

✨ AI Chat Integration — Uses Google Gemini (via OpenAI-compatible API) to understand natural shopping queries.  
🛒 Product Fetching — Retrieves real product listings from a live API (`template-03-api.vercel.app`).  
📦 Smart Responses — Answers questions about stock, prices, shipping, and more.  
🧠 Custom Tools — Includes function tools like `get_products()` and can be extended for shipping, returns, or order tracking.  
⚙️ Framework Ready — Built on the `agents` library for structured, multi-tool reasoning.  
💬 Rich CLI Output — Uses `rich` for colorful, easy-to-read console logs.  

---

🧩 Tech Stack

- Language: Python 3.10+  
- AI Model: Gemini 2.0 Flash  
- Framework: `agents`  
- Libraries:  
  - `requests`  
  - `python-dotenv`  
  - `rich`  

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/shopping-agent.git
cd shopping-agent
