import asyncio
import dotenv
import os
import sys
from micro_graph import Node, OutputWriter
from micro_graph.io.cli_input import ConsoleInputNode
from micro_graph.ai.llm_generation import LLMGenerateNode
from micro_graph.ai.llm import LLM
from micro_graph.ai.automatic_refinement_feedback_loop import automatic_refinement_feedback_loop

PLANNER = """Think carefully about the tasks required to fulfill the user's request.
Create a task list in markdown format that outlines the necessary steps.
If certain tasks must be completed in a specific order, make sure this order is clear in your list.
If there are subtasks, use nested lists in your markdown to show their relationship.

If a plan is already provided, review it and use it as a basis for your new plan.
Include any tasks from the current plan that are still relevant, as your new plan will replace the previous one.

---
Example Output:

* [ ] Check weather:
    - [ ] Find the current location of the user
    - [ ] Use the location to find the current weather
* [ ] Decide if you need nothing, a jacket or an umbrella

---
Query:
```
{query}
```

---
Current Plan (or empty):
```
{plan}
```

---
Feedback (or empty):
```
{feedback}
```
"""

PLANNING_FEEDBACK = """Given the following plan, provide constructive feedback.

- In the first few feedback rounds, focus on broad, high-level aspects of the plan.
- As iterations progress, shift your attention to more specific details and refinements.
- After several rounds of detailed feedback, if only minor issues remain, be supportive and offer positive feedback, indicating that the solution is solid.
- Ending with positive feedback helps boost morale and signals completion.

* Your primary goal is to ensure the plan fully addresses the user's intent.
* Avoid repeating feedback—do not mention the same point multiple times.
* Do not get stuck on minor details; exact wording is not critical.
* Assume that those executing the plan have reasonable intelligence and judgment.
* You have {max_iter} feedback rounds in total. This is iteration {iter}.

When giving feedback, use these principles:
1. Observation: Describe what you noticed or observed.
2. Impact: Explain how it affects you or the outcome.
3. Wish: State what you would like to see changed or kept the same.
4. Proposal: Suggest a concrete improvement or provide an example to fulfill your wish.

Write your feedback as if you are writing it for yourself.
---
Example (improvement suggestions):

* I notice there is no proof reading of the letter planned. Spelling errors could potentially slip through. I should plan time for proof reading.
* There are no figures in the document. This makes the document look unprofessional. Figures need to be added to the document.


Example (no more changes needed):

* The plan is good. No further iteration needed.

---
Query from User:
```
{query}
```

---
Plan ({iter}/{max_iter}):
```
{plan}
```

---
Prior Feedback:
```
{old_feedback}
```
"""


def planner_agent(llm: LLM, model: str, max_iterations: int = 5) -> Node:
    planner = LLMGenerateNode(llm=llm, model=model, prompt_template=PLANNER, field="plan", shared=True, output="thought")
    plan = automatic_refinement_feedback_loop(
        node=planner,
        llm=llm,
        model=model,
        feedback_template=PLANNING_FEEDBACK,
        max_iterations=max_iterations,
    )
    return plan


async def main():
    shared = {
        "query": "",
        "plan": "",
    }

    # LLM connection
    llm = LLM(
        api_endpoint=os.environ.get("API_ENDPOINT", "http://localhost:11434"),
        api_key=os.environ.get("API_KEY", "ollama"),
        provider=os.environ.get("PROVIDER", "ollama"),
        model=os.environ.get("MODEL", "AUTODETECT")
    )
    model = os.environ.get("MODEL", "gemma3:12b")

    # Create nodes for PRD generation
    if len(sys.argv) > 2:
        query_text = sys.argv[2:]
        shared["query"] = " ".join(query_text)
        query = Node()
    else:
        query = ConsoleInputNode(question="Query", field="query", shared=True)
    
    plan = planner_agent(llm, model=model, max_iterations=3)
    query.then(plan)

    # Run the flow starting with get_idea
    output = OutputWriter()
    await query(output, shared)
    print(shared["plan"])

if __name__ == "__main__":
    dotenv.load_dotenv()
    asyncio.run(main())
