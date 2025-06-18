# agents.py
from autogen import (
    AssistantAgent, 
    ConversableAgent,
    GroupChat, 
    GroupChatManager,
    UserProxyAgent
)

from config import make_llm_config, FREE_MODELS
from utils import load_prompt

def create_llm_configs(llm_configs=None):
    """Create a dictionary of LLM configurations."""
    return {
        "coding": make_llm_config(FREE_MODELS["coding"]),
        "reasoning": make_llm_config(FREE_MODELS["reasoning"]),
        "general": make_llm_config(FREE_MODELS["general"]),
    }

def create_genetic_optimizer_agent(llm_config):
    """Create an agent specialized for genetic algorithm optimization"""
    return ConversableAgent(
        name="GeneticOptimizer",
        system_message="""You are a genetic algorithm optimization specialist.
        Your role is to:
        1. Generate diverse code mutations
        2. Perform intelligent code crossover
        3. Suggest performance optimizations
        4. Evaluate code quality improvements
        
        Always focus on maintaining functionality while exploring improvements.""",
        llm_config=llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3
    )

def create_task_analyzer_agent(llm_config):
    """Creates the Task Analyzer agent with web search capabilities."""
    system_message = load_prompt("task_analyzer")
    if not system_message:
        system_message = "You are a task analysis specialist. Analyze programming tasks and recommend strategies."
    return AssistantAgent(
        name="TaskAnalyzer",
        llm_config=llm_config,
        system_message=system_message,
    )

def create_all_agents(llm_configs=None):
    """Creates and returns a dictionary of all required agents, including the genetic optimizer."""
    llm_configs = create_llm_configs()
    
    task_analyzer = create_task_analyzer_agent(llm_configs["reasoning"])
    codegen = AssistantAgent(
        name="CodeGen", 
        llm_config=llm_configs["coding"], 
        system_message=load_prompt("codegen")
    )
    reasoner = AssistantAgent(
        name="LogicalReasoner", 
        llm_config=llm_configs["reasoning"], 
        system_message="You are a logical reasoning specialist..."
    )
    testwriter = AssistantAgent(
        name="TestCaseWriter", 
        llm_config=llm_configs["coding"], 
        system_message=load_prompt("testcase")
    )
    corrector = ConversableAgent(
        name="Corrector", 
        llm_config=llm_configs["coding"], 
        system_message=load_prompt("corrector"), 
        human_input_mode="NEVER"
    )
    genetic_optimizer = create_genetic_optimizer_agent(llm_configs["reasoning"])
    
    user_proxy = UserProxyAgent(
        name="UserProxy",
        human_input_mode="NEVER",
        code_execution_config=False,
    )

    return {
        "task_analyzer": task_analyzer,
        "codegen": codegen,
        "reasoner": reasoner,
        "testwriter": testwriter,
        "corrector": corrector,
        "genetic_optimizer": genetic_optimizer,
        "user_proxy": user_proxy,
    }
