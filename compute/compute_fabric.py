from typing import Dict, Any, Optional, Callable, List
import asyncio
import logging
from dataclasses import dataclass
from collections import deque
import uuid
import time
from skills.todo_skill import ToDoSkill

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/compute_fabric.log")
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BackendInfo:
    backend: Any
    name: str
    priority: int
    health: float = 1.0
    last_used: float = 0.0
    max_concurrent: int = 5
    failure_count: int = 0

@dataclass
class TaskTelemetry:
    task_id: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "pending"
    backend: Optional[str] = None
    error: Optional[str] = None

class DummyBackend:
    def __init__(self, name: str = "DummyBackend"):
        self.name = name
        self.semaphore = asyncio.Semaphore(5)
        self.todo_skill = ToDoSkill()

    async def run(self, task: Dict) -> Any:
        """Handle tasks, integrating with ToDoSkill."""
        async with self.semaphore:
            try:
                logger.info(f"{self.name} running task: {task}")
                skill = task.get("skill")
                action = task.get("action")
                params = task.get("params", {})
                if skill == "todo_skill":
                    if action == "add":
                        result = self.todo_skill.add_todo(**params)
                        return result if result else f"Failed to add todo: {params}"
                    elif action == "list":
                        return self.todo_skill.list_todos()
                    elif action == "done":
                        success = self.todo_skill.mark_done(**params)
                        return f"Marked todo {params.get('todo_id')} as done" if success else f"Failed to mark todo {params.get('todo_id')}"
                    elif action == "remove":
                        success = self.todo_skill.remove_todo(**params)
                        return f"Removed todo {params.get('todo_id')}" if success else f"Failed to remove todo {params.get('todo_id')}"
                await asyncio.sleep(0.05)  # Simulate processing
                return f"{self.name} ran task: {task}"
            except Exception as e:
                logger.error(f"{self.name} failed task {task}: {e}")
                raise

    async def health_check(self) -> float:
        """Simulate backend health check."""
        try:
            await asyncio.sleep(0.01)
            return 1.0
        except Exception as e:
            logger.error(f"Health check failed for {self.name}: {e}")
            return 0.0

class ComputeFabric:
    def __init__(self):
        self.backends: Dict[str, BackendInfo] = {
            "dummy": BackendInfo(backend=DummyBackend("DefaultDummy"), name="DefaultDummy", priority=1)
        }
        self.task_queue = deque()
        self.max_retries = 3
        self.retry_delay = 0.2
        self.telemetry: List[TaskTelemetry] = []
        self.fallback_handler: Callable[[Dict], Any] = lambda task: f"Fallback: Unhandled task {task}"
        self._start_monitoring()

    def _start_monitoring(self) -> None:
        """Start background health monitoring."""
        async def monitor():
            while True:
                for backend_info in self.backends.values():
                    backend_info.health = await backend_info.backend.health_check()
                    logger.debug(f"Health check for {backend_info.name}: {backend_info.health}")
                await asyncio.sleep(30)  # Check every 30 seconds
        asyncio.create_task(monitor())

    def register_backend(self, name: str, backend: Any, priority: int = 1, max_concurrent: int = 5) -> None:
        """Register backend with priority and concurrency limit."""
        self.backends[name] = BackendInfo(backend=backend, name=name, priority=priority, max_concurrent=max_concurrent)
        logger.info(f"Registered backend: {name}, priority: {priority}")

    def register_fallback(self, handler: Callable[[Dict], Any]) -> None:
        """Set custom fallback handler."""
        self.fallback_handler = handler
        logger.info("Registered fallback handler")

    async def _select_backend(self, prefer_quantum: bool) -> Optional[str]:
        """Select best backend based on priority, health, and load."""
        healthy_backends = [
            info for info in self.backends.values()
            if info.health > 0.7 and info.failure_count < 5
        ]
        if not healthy_backends:
            logger.warning("No healthy backends available")
            return None
        if prefer_quantum:
            quantum_backends = [info for info in healthy_backends if "quantum" in info.name.lower()]
            if quantum_backends:
                return min(quantum_backends, key=lambda x: (x.last_used, x.failure_count, -x.priority)).name
        return min(healthy_backends, key=lambda x: (x.last_used, x.failure_count, -x.priority)).name

    async def dispatch_task(self, task: Dict, prefer_quantum: bool = False, priority: int = 1) -> Any:
        """Dispatch task to best backend with retries and telemetry."""
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        telemetry = TaskTelemetry(task_id=task_id, start_time=time.time())
        self.telemetry.append(telemetry)

        self.task_queue.append((priority, task_id, task))
        self.task_queue = deque(sorted(self.task_queue, key=lambda x: x[0], reverse=True))

        while self.task_queue:
            _, current_task_id, current_task = self.task_queue.popleft()
            for attempt in range(self.max_retries):
                try:
                    backend_name = await self._select_backend(prefer_quantum)
                    if not backend_name:
                        telemetry.status = "failed"
                        telemetry.error = "No healthy backends"
                        return self.fallback_handler(current_task)
                    backend_info = self.backends[backend_name]
                    telemetry.backend = backend_name
                    backend_info.last_used = time.time()
                    result = await backend_info.backend.run(current_task)
                    telemetry.end_time = time.time()
                    telemetry.status = "success"
                    logger.info(f"Task {current_task_id} completed by {backend_name}")
                    return result
                except Exception as e:
                    backend_info.failure_count += 1
                    logger.error(f"Attempt {attempt + 1}/{self.max_retries} failed for task {current_task_id}: {e}")
                    if attempt + 1 == self.max_retries:
                        telemetry.status = "failed"
                        telemetry.error = str(e)
                        return self.fallback_handler(current_task)
                    await asyncio.sleep(self.retry_delay)
        return None

    def sync_dispatch_task(self, task: Dict, prefer_quantum: bool = False, priority: int = 1) -> Any:
        """Synchronous wrapper for dispatch_task."""
        return asyncio.run(self.dispatch_task(task, prefer_quantum, priority))

    def get_telemetry(self) -> List[Dict]:
        """Return telemetry data."""
        return [
            {
                "task_id": t.task_id,
                "duration": (t.end_time - t.start_time) if t.end_time else None,
                "status": t.status,
                "backend": t.backend,
                "error": t.error
            }
            for t in self.telemetry
        ]