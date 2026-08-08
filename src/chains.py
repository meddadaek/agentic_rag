from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from src.config import LLM_MODEL, TEMPERATURE


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


llm = ChatGroq(model=LLM_MODEL, temperature=TEMPERATURE)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

SYS_PROMPT_GRADER = """You are an expert grader assessing relevance of a retrieved document to a user question.
Follow these instructions for grading:
  - If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
  - Your grade should be either 'yes' or 'no' to indicate whether the document is relevant to the question or not."""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", SYS_PROMPT_GRADER),
    ("human", """Retrieved document:
{document}
User question:
{question}
"""),
])

doc_grader = grade_prompt | structured_llm_grader


def format_docs(docs):
    """Join multiple documents with double newlines for the prompt context."""
    if isinstance(docs, str):
        return docs
    return "\n\n".join(doc.page_content for doc in docs)


rag_prompt_template = """
you are an assistant for question-answering tasks. use the following pieces
of retrived context to answer the question .if no context is present or if you dont know to answer the question
just say "I don't know" . do not make up an answer unless it is there in the context provided
give a detailed answer and to the point answer with regard to the question

Question :
{question}
=========
context:
{context}


answer :


"""

prompt_template = ChatPromptTemplate.from_template(rag_prompt_template)
chat_llm = ChatGroq(model=LLM_MODEL, temperature=TEMPERATURE)

qa_rag_chain = (
    {
        "context": itemgetter("context") | RunnableLambda(format_docs),
        "question": itemgetter("question"),
    }
    | prompt_template
    | chat_llm
    | StrOutputParser()
)


SYS_PROMPT_REWRITING = """
act as a question re-writer and perform the foloowing task :
- convert the following input question to a better vesrion that is optimized for web search 
- when re-writing look at the input question and try to reason about the underlying semantic intent /meaning
"""


re_write_prompt = ChatPromptTemplate.from_messages([
    ("system", SYS_PROMPT_REWRITING),
    ("human", """here is the initial question :
     {question}
     formulate an improved question"""),
])


question_rewriter = re_write_prompt | chat_llm | StrOutputParser()
