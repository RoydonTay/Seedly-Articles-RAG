import langchain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp

langchain.debug = True
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="/home/cowboygarage/seedly_scrape/llama-2-7b-chat.Q4_K_M.gguf",
    temperature=0.1,
    max_tokens=2000,
    top_p=1,
    n_ctx = 800,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

prompt = """
Question: What should I consider when getting insurance?
"""
llm(prompt)