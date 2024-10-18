from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class GradeAnswer(BaseModel):

    binary_score: bool = Field(description="Answer the question, 'yes' or 'no'")


structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """
You are a grader assessing whether an answer addresses / resolves a question. \n
Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question.
"""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader