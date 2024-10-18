import pytest
from dotenv import load_dotenv

load_dotenv()

from crag.graph.chains.generation import generation_chain
from crag.ingestion import retriever
from srag.graph.chains.hallucination_grader import (
    GradeHallucinations,
    hallucination_grader,
)


def test_hallucination_grader_answer_yes():
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score


def test_hallucination_grader_answer_no():
    question = "agent memory"
    docs = retriever.invoke(question)

    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": "How do you make pizza?"}
    )
    assert not res.binary_score


if __name__ == "__main__":
    pytest.main(["-s -v", __file__])
