from skills.todo_skill import ToDoSkill

class SkillTree:
    def __init__(self):
        self.skills = {'todo': ToDoSkill()}

    def get_skill(self, skill_name: str):
        return self.skills.get(skill_name)

    async def process_intent(self, intent_data: dict) -> dict:
        skill_name = 'todo'  # Simplified for launch
        skill = self.get_skill(skill_name)
        if skill:
            return await skill.process_intent(intent_data)
        return {'status': 'error', 'message': f'Skill {skill_name} not found'}
