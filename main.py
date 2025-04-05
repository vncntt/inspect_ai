import re
from inspect_ai import Task, task, Epochs
from inspect_ai.model import get_model
from inspect_ai.dataset import example_dataset, hf_dataset, FieldSpec
from inspect_ai.scorer import model_graded_fact, at_least, CORRECT, INCORRECT, Score, AnswerPattern, stderr, scorer, accuracy, Target
from inspect_ai.solver import generate, TaskState, prompt_template
from inspect_ai import eval


math_dataset = hf_dataset(
    "HuggingFaceH4/MATH-500",
    split="test",
    sample_fields=FieldSpec(input="problem", target="solution"),
    shuffle=False,
    trust=True,
)
PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line
of your response should be of the form ANSWER: $ANSWER (without
quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line after "ANSWER:",
and you do not need to use a \\boxed command.
""".strip()

EQUIVALENCE_TEMPLATE = r"""
Look at the following two expressions (answers to a math problem)
and judge whether they are equivalent. Only perform trivial 
simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)
---

YOUR TASK

Respond with only "Yes" or "No" (without quotes). Do not include
a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s
""".strip()
@task
def fail_at_k(k: int):
    return Task(
        dataset=math_dataset,
        solver=[generate(prompt_template=PROMPT_TEMPLATE)],
        scorer=model_graded_fact(),
        epochs=Epochs(k, at_least(k))  
    )

@task
def accuracy():
    return Task(
        dataset=math_dataset,
        solver=[generate(prompt_template=PROMPT_TEMPLATE)],
        scorer=model_graded_fact(),
    )

@task
def pass_at_k(k: int):
    return Task(
        dataset=math_dataset,
        solver=[generate(prompt_template=PROMPT_TEMPLATE)],
        scorer=model_graded_fact(),
        epochs=Epochs(k, at_least(1))  
    )


# eval(fail_at_k(3), model="openrouter/anthropic/claude-3.7-sonnet", limit=10)
# eval(accuracy(), model="openrouter/anthropic/claude-3.7-sonnet", limit=10)
eval(pass_at_k(3), model="openrouter/anthropic/claude-3.7-sonnet", limit=10)
