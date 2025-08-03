from micro_graph import Node, NodeResult, OutputWriter, template_formatting


class ConsoleInputNode(Node):
    """
    Node for micro-graph that prompts the user for input in the console.
    
    Returns:
        The kwargs and additionally the user input stored in the specified `field`.
        If shared is true, the user input is stored in the shared object instead of returning it.
    """

    def __init__(self, question: str = "User", field: str = "user_input", shared: bool = False):
        super().__init__()
        self.question = question
        self.field = field
        self.shared = shared

    async def run(self, output: OutputWriter, shared: dict, **kwargs) -> NodeResult:
        user_input = input(template_formatting(self.question, shared, **kwargs) + ' > ')
        if self.shared:
            shared[self.field] = user_input
            return kwargs
        else:
            return {self.field: user_input, **kwargs}
