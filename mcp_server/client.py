from mcp import ClientSession, StdioServerParameters
from langchain_openai import ChatOpenAI
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
import asyncio
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

server_params = StdioServerParameters(
    command="python", 
    args=["server.py"]
)

async def run_agent(): 
    async with stdio_client(server_params) as (read, write): 
        async with ClientSession(read, write) as session: 
            await session.initialize()
            tools=await load_mcp_tools(session)
            agent = create_react_agent(model, tools)
            agent_response = await agent.ainvoke(
                {"messages": "what is 5 + 6?"}
            )
            return agent_response["messages"][3].content
        
if __name__ == "__main__": 
    result = asyncio.run(run_agent())
    print(result)