# from bigcode_eval.models.base import BaseModel
from socratic_core import SocraticCodeGenerator

class BaseModel:
    """Dummy base class so Socratic works without bigcode_eval.models.base"""
    pass

class Socratic(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.generator = SocraticCodeGenerator()

    def generate(self, prompts, max_length, temperature, **kwargs):
        outputs = []

        for prompt in prompts:
            problem = {
                "id": "humaneval",
                "title": "",
                "description": prompt
            }
            results, _ = self.generator.generate_for_problem(problem)
            outputs.append(results["code"])

        return outputs
