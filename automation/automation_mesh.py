from typing import List, Dict, Any, Optional, Callable
import asyncio
import logging
import time
from dataclasses import dataclass
from collections import defaultdict
from functools import lru_cache
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("automation_mesh.log")]
)
logger = logging.getLogger(__name__)

@dataclass
class AutomatorInfo:
    automator: Any
    priority: int  # Higher = preferred
    name: str
    max_concurrent: int = 5  # Max concurrent tasks per automator
    failure_count: int = 0  # Track failures for load balancing

@dataclass
class Telemetry:
    step_id: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "pending"
    error: Optional[str] = None

class DummyAutomator:
    def __init__(self, name: str = "DummyAutomator"):
        self.name = name

    @lru_cache(maxsize=100)
    async def can_handle(self, step_key: str) -> bool:
        """Cached check for step handling."""
        try:
            step = eval(step_key)  # Convert stringified step back to dict
            skill = step.get("skill", "")
            action = step.get("action", "")
            return skill in ["todo_skill", "dummy_skill"] and action in ["add", "list", "done", "remove", "execute"]
        except Exception as e:
            logger.error(f"{self.name} failed to evaluate step: {e}")
            return False

    async def handle(self, step: Dict) -> str:
        """Handle step with simulated processing."""
        try:
            await asyncio.sleep(0.05)  # Simulate async work
            return f"{self.name} handled step: {step}"
        except Exception as e:
            logger.error(f"{self.name} failed to handle step {step}: {e}")
            raise

class AutomationMesh:
    def __init__(self):
        self.automators: List[AutomatorInfo] = [
            AutomatorInfo(automator=DummyAutomator("DefaultDummy"), priority=1, name="DefaultDummy")
        ]
        self.max_retries = 3
        self.retry_delay = 0.2
        self.semaphore: Dict[str, asyncio.Semaphore] = defaultdict(lambda: asyncio.Semaphore(5))
        self.telemetry: List[Telemetry] = []
        self.fallback_handler: Callable[[Dict], str] = lambda step: f"Fallback: Unhandled step {step}"

    def register(self, automator: Any, priority: int = 1, name: Optional[str] = None, max_concurrent: int = 5) -> None:
        """Register automator with priority and concurrency limit."""
        name = name or f"Automator_{uuid.uuid4().hex[:8]}"
        self.automators.append(AutomatorInfo(automator, priority, name, max_concurrent))
        self.automators.sort(key=lambda x: (x.priority, -x.failure_count), reverse=True)
        self.semaphore[name] = asyncio.Semaphore(max_concurrent)
        logger.info(f"Registered {name} with priority {priority}, max_concurrent {max_concurrent}")

    def register_fallback(self, handler: Callable[[Dict], str]) -> None:
        """Set custom fallback handler."""
        self.fallback_handler = handler
        logger.info("Registered custom fallback handler")

    async def execute_step(self, step: Dict, step_id: str) -> Any:
        """Execute a single step with retry and telemetry."""
        telemetry = Telemetry(step_id=step_id, start_time=time.time())
        self.telemetry.append(telemetry)
        
        for attempt in range(self.max_retries):
            try:
                # Serialize step for caching
                step_key = str(step)
                for automator_info in self.automators:
                    async with self.semaphore[automator_info.name]:
                        if await automator_info.automator.can_handle(step_key):
                            logger.info(f"{automator_info.name} handling step {step_id}")
                            result = await automator_info.automator.handle(step)
                            telemetry.end_time = time.time()
                            telemetry.status = "success"
                            return result
                logger.warning(f"No automator could handle step {step_id}")
                telemetry.status = "failed"
                telemetry.error = "No automator available"
                return self.fallback_handler(step)
            except Exception as e:
                automator_info.failure_count += 1  # Penalize failing automator
                self.automators.sort(key=lambda x: (x.priority, -x.failure_count), reverse=True)
                logger.error(f"Attempt {attempt + 1}/{self.max_retries} failed for step {step_id}: {e}")
                if attempt + 1 == self.max_retries:
                    telemetry.status = "failed"
                    telemetry.error = str(e)
                    return f"Error: Failed step {step_id} after {self.max_retries} attempts: {e}"
                await asyncio.sleep(self.retry_delay)
        return None

    async def execute(self, plan: List[Dict]) -> List[Any]:
        """Execute plan with batch concurrency and telemetry."""
        if not plan:
            logger.warning("Empty plan received")
            return []

        # Batch steps for parallel execution
        batch_size = min(len(plan), 10)  # Configurable batch size
        results = []
        for i in range(0, len(plan), batch_size):
            batch = plan[i:i + batch_size]
            tasks = [self.execute_step(step, f"step_{uuid.uuid4().hex[:8]}") for step in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            results.extend(batch_results)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Step {i} failed: {result}")
                results[i] = f"Error: {result}"

        logger.info(f"Plan execution completed with {len(results)} results")
        return results

    def sync_execute(self, plan: List[Dict]) -> List[Any]:
        """Synchronous wrapper for execute."""
        return asyncio.run(self.execute(plan))

    def get_telemetry(self) -> List[Dict]:
        """Return telemetry data for monitoring."""
        return [
            {
                "step_id": t.step_id,
                "duration": (t.end_time - t.start_time) if t.end_time else None,
                "status": t.status,
                "error": t.error
            }
            for t in self.telemetry
        ]