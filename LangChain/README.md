This repo comes up with two important scripts:
- **vectordbsetup.py**: it setups a vector DB containing LangChain documentation, used for possible RAG
- **routing.py**: it performs semantic routing deciding between RAG, in case of specific queries, and normal inference, in case of general queries.
To make it work, a TogetherAI API key is needed and it must be stored in a file called **config.ini** with the format below:

**[API]**  
**TOGETHER_API_KEY=your_together_ai_secret_key** 

The vector db and the necessary index are already available in the repository, as well as the LangChain documentation which has been indexed.
