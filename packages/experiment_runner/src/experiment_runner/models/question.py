from pydantic import BaseModel


class Question(BaseModel):
    id: str
    corpus: str
    level: int
    question: str
    expected_facts: list[str]
