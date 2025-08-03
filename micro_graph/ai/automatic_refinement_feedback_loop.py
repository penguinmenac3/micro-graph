from micro_graph import Node, NodeResult, OutputWriter
from micro_graph.ai.llm_generation import LLMGenerateNode, LLMDecisionNode
from micro_graph.ai.llm import LLM


def automatic_refinement_feedback_loop(node: Node, llm: LLM, model: str, feedback_template: str, max_iterations: int = 5):
    """
    Automatically refine the output of a node by recieving feedback from an LLM and iterating until the feedback is accepted or max_iterations is reached.
    """
    async def loop_node(output: OutputWriter, shared: dict, **kwargs) -> NodeResult:
        feedback_acceptance="You are a meta reviewer. Based on the feedback you decide if the result requires more iterations (reject) or if the result is already good enough and the feedback is just critiquing irrelevant details (accept).\n"\
            "Feedback:\n```\n{feedback}\n```\n\nWhat do you recommend? (accept/reject)"

        feedback = LLMGenerateNode(llm=llm, model=model, prompt_template=feedback_template, field="feedback")
        decision = LLMDecisionNode(llm=llm, model=model, prompt_template=feedback_acceptance, field="accept")
        
        node_result = None
        kwargs["feedback"] = ""
        old_feedback= ""
        for i in range(max_iterations):
            output.thought(f"Generating output (iteration {i + 1} of {max_iterations})")
            node_result = await node(output, shared, **kwargs)
            output.thought("Giving feedback on the output")
            feedback_result = await feedback(output, shared, iter=i+1, max_iter=max_iterations, old_feedback=old_feedback, **(node_result or {}))
            if feedback_result is not None:
                kwargs["feedback"] = feedback_result["feedback"]
                old_feedback += feedback_result["feedback"] + "\n"
            decision_result = await decision(output, shared, **(feedback_result or {}))
            if decision_result is not None and decision_result["accept"].lower() == "accept":
                break
            if i == max_iterations - 1:
                output.thought("Reached the maximum number of iterations. Returning the last result.")
        return node_result
    return Node(run=loop_node)
