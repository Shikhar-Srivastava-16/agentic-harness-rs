# Agentic AI in Rust

## Agent Orchestration

Agentic AI has two broad components: an *LLM* which acts like the 'brain', and a *Harness*, which makes a  requires a **Harness** in order to perform computation that cannot be done by the LLM

### Harness Features

A harness in an agentic AI system, or an **Agent Harness** as they are sometimes called, will typically need to perform a number of tasks. This includes:
 - **Context**: LLMs are *'stateless systems'*, which means that they have no native memory. Regardless of any convenience systems that providers may have built, harnesses maintain the history of a conversation and manage the roles of messages. All of this is done so that the LLM has access to all of the previous information in the conversation history. The part of the history which is passed into the system is known as the **Context Window**.
 - **Memory**: LLMs cannot maintain everything in context, so we store certain information on the runner device on the Agents. This is usually done through RAG systems. Harnesses typically provide access to these systems. 
 - **Tools**: The vast majority of tasks cannot be done well by LLMs. The majority of these tasks can also done much more efficiently and correctly using other computer systems. **Harnesses provide access and a runtime for these tools**.
 
This framework allows for all of these features to be added into an **agent harness** that is built natively in `rust`. - with the notable exception of the RAG-like systems, which must be provided by the user. The means that this framework provides everything you need in order to manage the prompts of the AI system. Anything that influences the content of those prompts must be provided. 
