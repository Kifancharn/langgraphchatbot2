from langchain_google_genai import ChatGoogleGenerativeAI
import os

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key="ADD API KEY")
result = llm.invoke("Sing a ballad of LangChain.")

print(result)
