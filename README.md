# Agentic AI in Rust

## Crate Features

The crate works using rust's monomorphic type system to format queries. Agentic Systems involve the repeated transfer of a very large amount of data. All of this data is heavily structured - it involves a JSON (which is semi-structured), with a rigid specification for what the JSON fields are expected to contain. All agentic systems can be said to have two broad categories of code, delieneated by what part of the data transfer it controls:
- **Structure**: Code that deals with formatting, structuring, and handling of the queries. This is highly structured code, and the input/output is highly predictable
- **Content**: Code that deals with the content of queries. This is highly dependent on the **purpose and design** of the individual agent.

Queries are the only means by which the LLM communicates with the harness. Due to the determinability of the *Structure* code, it can be heavily standardised. This crate will provide code that has boilerplate-free ways of the 

## Agent Orchestration

An AI Agent has two broad components: an *LLM* which acts like the 'brain', and a *Harness*, which does everything else. Given that LLMs are hyper-specialised to prediction and generation, they need other software to do different tasks. That means that they need a **Harness** to provide access to this extra software, in order to perform computation that cannot (or should not) be done by the LLM.

### Harness Features

A harness in an agentic AI system, or an **Agent Harness** as they are sometimes called, will typically need to perform a number of tasks. This includes:
 - **Context**: LLMs are *'stateless systems'*, which means that they have no inbuilt memory. Regardless of any convenience systems that providers may have built, harnesses maintain the history of a conversation and manage the roles of messages. All of this is done so that the LLM has access to all of the previous information in the conversation history. The part of the history which is passed into the system is known as the **Context Window**.
 - **Memory**: LLMs cannot maintain everything in context, so we store certain information on the runner device on the Agents. This is usually done through RAG systems. Harnesses typically provide access to these systems. 
 - **Tools**: The vast majority of tasks cannot be done well by LLMs. The majority of these tasks can also done much more efficiently and correctly using other computer systems. **Harnesses provide access and a runtime for these tools**.
 
This framework allows for all of these features to be added into an **agent harness** that is built natively in `rust`. - with the notable exception of the RAG-like systems, which must be provided by the user. The means that this framework provides everything you need in order to manage the prompts of the AI system. Anything that influences the content of those prompts must be provided by the user. All of the functionality that the harness provides is standardised using the trait system. It uses rust's advanced Polymorphism in order to send to models that have different input expectations. While the harness is not sending requests or recieving them from the LLM, it uses an internal representation to standardise the format of a prompt. 

It is designed to be very modular. This has a few implications:
- Adding a new LLM to the Framework is very simple: it simply involves adding a struct that can represent the input, and then adding logic to convert between the IR and the new struct.
