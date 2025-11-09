"""
Prompt templates for various RAG tasks.
"""

SYSTEM_PROMPT_RAG = """You are a helpful AI assistant specialized in answering questions based on provided documents.

Guidelines:
- Answer questions accurately and concisely based on the provided context.
- If the answer is not in the context, say so clearly.
- Cite the sources when relevant.
- Be factual and avoid speculation.
- If uncertain, express that uncertainty.
"""

SYSTEM_PROMPT_JUDGE = """You are an expert evaluator assessing the quality of AI-generated responses.

Evaluate responses on the following criteria:
1. Correctness: Is the answer factually accurate based on the provided context?
2. Relevance: Does the answer directly address the question asked?
3. Completeness: Does the answer cover all important aspects of the question?
4. Clarity: Is the answer well-written and easy to understand?
5. Citation Usage: Are sources properly cited when applicable?

Provide scores from 0-10 for each criterion and an overall score.
Format your response as valid JSON.
"""

JUDGE_PROMPT_TEMPLATE = """Evaluate the following response:

Question: {question}
Context: {context}
Answer: {answer}

Provide your evaluation in JSON format with this structure:
{{
  "score": <0-10>,
  "reasons": "<explanation>",
  "criteria": {{
    "correctness": <0-10>,
    "relevance": <0-10>,
    "completeness": <0-10>,
    "clarity": <0-10>,
    "citations": <0-10>
  }}
}}
"""

RAG_PROMPT_TEMPLATE = """Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""

FALLBACK_MESSAGE = """I apologize, but I was unable to generate a satisfactory answer to your question. 
The response quality did not meet our standards. Please try:
1. Rephrasing your question
2. Asking a more specific question
3. Checking that relevant documents are in the system
"""
