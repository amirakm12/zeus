$todoSkillContent = @"
class ToDoSkill:
    def __init__(self):
        self.todos = []
        self.next_id = 1

    async def process_intent(self, intent_data: dict) -> dict:
        intent = intent_data.get('content', '').split(':', 1)[0].strip()
        content = intent_data.get('content', '').split(':', 1)[1].strip() if ':' in intent_data.get('content', '') else ''

        if intent == 'Add todo':
            todo = {'id': self.next_id, 'content': content, 'done': False}
            self.todos.append(todo)
            self.next_id += 1
            return {'status': 'success', 'message': f'Added todo: {content}', 'todo': todo}
        elif intent == 'List todos':
            return {'status': 'success', 'todos': self.todos}
        elif intent == 'Mark done':
            try:
                todo_id = int(content)
                for todo in self.todos:
                    if todo['id'] == todo_id:
                        todo['done'] = True
                        return {'status': 'success', 'message': f'Marked todo {todo_id} as done'}
                return {'status': 'error', 'message': f'Todo {todo_id} not found'}
            except ValueError:
                return {'status': 'error', 'message': 'Invalid todo ID'}
        elif intent == 'Remove todo':
            try:
                todo_id = int(content)
                for todo in self.todos:
                    if todo['id'] == todo_id:
                        self.todos.remove(todo)
                        return {'status': 'success', 'message': f'Removed todo {todo_id}'}
                return {'status': 'error', 'message': f'Todo {todo_id} not found'}
            except ValueError:
                return {'status': 'error', 'message': 'Invalid todo ID'}
        return {'status': 'error', 'message': 'Unknown intent'}
"@
$todoSkillContent | Out-File -FilePath "todo_skill.py" -Encoding utf8