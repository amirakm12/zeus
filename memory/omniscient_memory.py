class OmniscientMemory:
    def __init__(self):
        self.knowledge_graph = {}

    def store(self, interaction):
        key = hash(str(interaction))
        self.knowledge_graph[key] = interaction

    def retrieve(self, query):
        return list(self.knowledge_graph.values())