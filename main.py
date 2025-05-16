# main.py

#pip install openai python-dotenv PyYAML faiss-cpu scikit-learn pandas sqlite3 tiktoken langgraph langchain_core

import os
from dotenv import load_dotenv
from app_graph import app # Import the compiled LangGraph app
from graph_state import AgentState
from utils import logger, load_agent_registry

def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        logger.critical("OPENAI_API_KEY not set. Exiting.")
        return

    # Initialize agent registry (primarily for meta-queries to inspect it)
    load_agent_registry() 
    logger.info("AI Customer Support Assistant (LangGraph Version)")
    logger.info("Type 'exit' or 'quit' to end.")

    # Optional: Load conversation history if implementing
    # history = []

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                logger.info("Exiting assistant.")
                break
            if not user_input.strip():
                continue

            initial_state: AgentState = {
                "original_query": user_input,
                "intent": None,
                "entities": None,
                "sql_query_generated": None,
                "sql_query_result": None,
                "retrieved_contexts": None,
                "rag_summary": None,
                "intermediate_response": None,
                "final_answer": None,
                "error_message": None,
                "history": [], # Add current interaction to history if used
                "processing_steps_versions": {}
            }
            
            # Stream events from the graph
            # for event in app.stream(initial_state):
            #     for key, value in event.items():
            #         logger.debug(f"Graph Event - Node: {key}, Output: {value}")
            # final_state = event[list(event.keys())[-1]] # Get the output of the last node (END)

            # Or invoke for a single final result
            final_state = app.invoke(initial_state)

            if final_state:
                if final_state.get("final_answer"):
                    print(f"Assistant: {final_state['final_answer']}")
                elif final_state.get("error_message"):
                    print(f"Assistant: I encountered an error: {final_state['error_message']}")
                else:
                    print("Assistant: I'm not sure how to respond to that.")
                
                logger.debug(f"Final State: {final_state}")
            else:
                print("Assistant: Sorry, something went wrong and I couldn't process your request.")

        except KeyboardInterrupt:
            logger.info("\nExiting assistant (Ctrl+C).")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
            print("Assistant: I'm sorry, a critical error occurred.")

if __name__ == "__main__":
    main()