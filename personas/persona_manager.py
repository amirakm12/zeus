from typing import Dict, List, Optional

class Persona:
    def __init__(self, name: str, specialties: List[str]):
        self.name = name
        self.specialties = specialties

    def interpret(self, input_data: Dict) -> Dict:
        return {"intent": input_data["content"], "context": self.specialties}

class PersonaManager:
    def __init__(self):
        self.personas: Dict[str, Persona] = {}

    def create_persona(self, name: str, specialties: List[str]) -> Persona:
        persona = Persona(name, specialties)
        self.personas[name] = persona
        return persona

    def choose_best_persona(self, multimodal_input: Dict) -> Optional[Persona]:
        if not self.personas:
            print("No personas available")
            return None
        return next(iter(self.personas.values()))

    def interpret(self, multimodal_input: Dict) -> Optional[Dict]:
        persona = self.choose_best_persona(multimodal_input)
        if persona is None:
            return None
        return persona.interpret(multimodal_input)