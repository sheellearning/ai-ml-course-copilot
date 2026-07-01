import os

from langchain_openai import ChatOpenAI


# Confirm that the API key is available without displaying it.
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY was not found.\n"
        "Configure the environment variable and restart VS Code."
    )


# Send one small request to confirm the OpenAI connection.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

response = llm.invoke(
    "Reply with exactly: OK - OpenAI connection successful"
)

print(response.content)
