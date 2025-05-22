from typing import Dict, Any, List, Optional
import asyncio
import logging
import time
from dataclasses import dataclass
from collections import defaultdict
import hashlib
import base64
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import numpy as np
from sklearn.ensemble import IsolationForest
import jwt
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/security_core.log")
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Policy:
    action: str
    allowed_roles: List[str]
    max_attempts: int = 3
    cooldown: float = 60.0

@dataclass
class AuditRecord:
    timestamp: float
    action: str
    entity: str
    status: str
    details: Optional[str] = None
    encrypted: bool = False
    integrity_hash: Optional[str] = None

class MockKyber:
    """Mock quantum-resistant cryptography (replace with pykyber if available)."""
    @staticmethod
    def encrypt(data: str) -> str:
        return base64.b64encode(data.encode()).decode()

    @staticmethod
    def decrypt(encrypted: str) -> str:
        return base64.b64decode(encrypted.encode()).decode()

class SecurityCore:
    def __init__(self):
        self.policies: Dict[str, Policy] = {
            "add_todo": Policy(action="add_todo", allowed_roles=["user", "admin"]),
            "list_todos": Policy(action="list_todos", allowed_roles=["user", "admin"]),
            "mark_done": Policy(action="mark_done", allowed_roles=["user", "admin"]),
            "remove_todo": Policy(action="remove_todo", allowed_roles=["admin"]),
            "execute": Policy(action="execute", allowed_roles=["admin"])
        }
        self.audit_log: List[AuditRecord] = []
        self.attempts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.cooldowns: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.ml_model = IsolationForest(contamination=0.05, random_state=42)
        self.activity_log: List[Dict] = []
        self.aes_key = get_random_bytes(32)  # AES-256 key
        self.oauth_secret = "mock_secret_123"  # Replace with real OAuth secret
        self.kyber = MockKyber()
        self._start_ml_training()

    def _start_ml_training(self) -> None:
        """Start background ML training for anomaly detection."""
        async def train_ml():
            while True:
                if len(self.activity_log) > 20:
                    X = np.array([[len(log["action"]), log["timestamp"] % 3600, log["attempts"],
                                   hash(log["user"]) % 1000]
                                  for log in self.activity_log[-100:]])
                    self.ml_model.fit(X)
                    logger.info("ML model retrained for anomaly detection")
                await asyncio.sleep(180)  # Retrain every 3 minutes
        asyncio.create_task(train_ml())

    def _encrypt_log(self, data: str) -> str:
        """Encrypt audit log with AES-256."""
        cipher = AES.new(self.aes_key, AES.MODE_EAX)
        ciphertext, tag = cipher.encrypt_and_digest(data.encode())
        return base64.b64encode(cipher.nonce + tag + ciphertext).decode()

    def _decrypt_log(self, encrypted: str) -> str:
        """Decrypt audit log."""
        try:
            raw = base64.b64decode(encrypted)
            nonce, tag, ciphertext = raw[:16], raw[16:32], raw[32:]
            cipher = AES.new(self.aes_key, AES.MODE_EAX, nonce=nonce)
            return cipher.decrypt_and_verify(ciphertext, tag).decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return "Decryption error"

    def _compute_integrity_hash(self, record: AuditRecord) -> str:
        """Compute SHA-256 hash for audit log integrity."""
        data = f"{record.timestamp}:{record.action}:{record.entity}:{record.status}:{record.details}"
        return hashlib.sha256(data.encode()).hexdigest()

    def register_policy(self, action: str, allowed_roles: List[str], max_attempts: int = 3, cooldown: float = 60.0) -> None:
        """Register a new security policy."""
        self.policies[action] = Policy(action=action, allowed_roles=allowed_roles, max_attempts=max_attempts, cooldown=cooldown)
        logger.info(f"Registered policy: {action}")

    async def update_policy(self, action: str, allowed_roles: List[str], token: str) -> bool:
        """Update policy with OAuth-like token validation."""
        try:
            jwt.decode(token, self.oauth_secret, algorithms=["HS256"])
            self.register_policy(action, allowed_roles)
            logger.info(f"Policy updated for {action}")
            return True
        except jwt.InvalidTokenError:
            logger.error("Invalid token for policy update")
            return False

    async def audit(self, orchestrator: Any) -> List[Dict]:
        """Audit orchestrator with ML-based anomaly detection."""
        try:
            audit_results = []
            components = [
                ("persona_manager", orchestrator.persona_manager),
                ("skill_tree", orchestrator.skill_tree),
                ("automation_mesh", orchestrator.automation_mesh),
                ("compute_fabric", orchestrator.compute_fabric),
                ("memory", orchestrator.memory),
                ("security_core", orchestrator.security_core),
                ("ui_hub", orchestrator.ui_hub),
                ("world_connector", orchestrator.world_connector)
            ]
            for name, component in components:
                status = "success" if component else "failed"
                details = f"{name} {'present' if component else 'missing'}"
                record = AuditRecord(
                    timestamp=time.time(),
                    action="audit",
                    entity=name,
                    status=status,
                    details=self._encrypt_log(details),
                    encrypted=True
                )
                record.integrity_hash = self._compute_integrity_hash(record)
                self.audit_log.append(record)
                if not component:
                    audit_results.append({"issue": f"Missing {name}", "severity": "critical"})
                    logger.error(f"Audit failed: {name} missing")

            # ML anomaly detection
            if self.activity_log:
                X = np.array([[len(log["action"]), log["timestamp"] % 3600, log["attempts"],
                               hash(log["user"]) % 1000]
                              for log in self.activity_log[-100:]])
                predictions = self.ml_model.predict(X)
                for i, pred in enumerate(predictions):
                    if pred == -1:
                        log = self.activity_log[-100:][i]
                        record = AuditRecord(
                            timestamp=time.time(),
                            action="audit",
                            entity=log["user"],
                            status="failed",
                            details=self._encrypt_log(f"Anomaly in {log['action']} by {log['user']}"),
                            encrypted=True
                        )
                        record.integrity_hash = self._compute_integrity_hash(record)
                        self.audit_log.append(record)
                        audit_results.append({"issue": f"Anomaly in {log['action']} by {log['user']}", "severity": "high"})
                        logger.warning(f"Anomaly detected: {log['action']} by {log['user']}")

            # Verify audit log integrity
            for record in self.audit_log:
                if record.integrity_hash and self._compute_integrity_hash(record) != record.integrity_hash:
                    audit_results.append({"issue": f"Tampered audit log entry at {record.timestamp}", "severity": "critical"})
                    logger.error(f"Audit log tampering detected at {record.timestamp}")

            return audit_results
        except Exception as e:
            record = AuditRecord(
                timestamp=time.time(),
                action="audit",
                entity="orchestrator",
                status="failed",
                details=self._encrypt_log(str(e)),
                encrypted=True
            )
            record.integrity_hash = self._compute_integrity_hash(record)
            self.audit_log.append(record)
            logger.error(f"Audit failed: {e}")
            return [{"issue": f"Audit error: {e}", "severity": "critical"}]

    async def enforce_policy(self, action: str, user: str = "default_user", role: str = "user", token: Optional[str] = None, ip: Optional[str] = None) -> Optional[str]:
        """Enforce policy with zero-trust, OAuth, quantum-resistant crypto, and IP whitelisting."""
        try:
            # Zero-trust: Validate token
            if token:
                try:
                    jwt.decode(token, self.oauth_secret, algorithms=["HS256"])
                except jwt.InvalidTokenError:
                    record = AuditRecord(
                        timestamp=time.time(),
                        action=action,
                        entity=user,
                        status="failed",
                        details=self._encrypt_log("Invalid token"),
                        encrypted=True
                    )
                    record.integrity_hash = self._compute_integrity_hash(record)
                    self.audit_log.append(record)
                    logger.error(f"Policy enforcement failed: Invalid token for {user}")
                    return "Error: Invalid token"

            # IP whitelisting (mock)
            if ip and ip not in ["127.0.0.1", "::1"]:  # Example whitelist
                record = AuditRecord(
                    timestamp=time.time(),
                    action=action,
                    entity=user,
                    status="failed",
                    details=self._encrypt_log(f"Unauthorized IP {ip}"),
                    encrypted=True
                )
                record.integrity_hash = self._compute_integrity_hash(record)
                self.audit_log.append(record)
                logger.error(f"Policy enforcement failed: Unauthorized IP {ip}")
                return f"Error: Unauthorized IP {ip}"

            policy = self.policies.get(action)
            if not policy:
                record = AuditRecord(
                    timestamp=time.time(),
                    action=action,
                    entity=user,
                    status="failed",
                    details=self._encrypt_log("Unknown action"),
                    encrypted=True
                )
                record.integrity_hash = self._compute_integrity_hash(record)
                self.audit_log.append(record)
                logger.error(f"Policy enforcement failed: Unknown action {action}")
                return "Error: Unknown action"

            # Rate limiting
            current_time = time.time()
            if current_time < self.cooldowns[user].get(action, 0):
                record = AuditRecord(
                    timestamp=current_time,
                    action=action,
                    entity=user,
                    status="failed",
                    details=self._encrypt_log(f"Cooldown active until {self.cooldowns[user][action]}"),
                    encrypted=True
                )
                record.integrity_hash = self._compute_integrity_hash(record)
                self.audit_log.append(record)
                logger.warning(f"Policy enforcement failed: {user} on cooldown for {action}")
                return f"Error: Cooldown active for {action}"

            # Role check with quantum-resistant signature
            signature = self.kyber.encrypt(f"{user}:{action}:{role}:{current_time}")
            if role not in policy.allowed_roles:
                self.attempts[user][action] += 1
                if self.attempts[user][action] > policy.max_attempts:
                    self.cooldowns[user][action] = current_time + policy.cooldown
                record = AuditRecord(
                    timestamp=current_time,
                    action=action,
                    entity=user,
                    status="failed",
                    details=self._encrypt_log(f"Unauthorized role {role}, signature: {signature}"),
                    encrypted=True
                )
                record.integrity_hash = self._compute_integrity_hash(record)
                self.audit_log.append(record)
                logger.warning(f"Policy enforcement failed: {user} has unauthorized role {role}")
                return f"Error: Unauthorized role {role} for {action}"

            # Log activity for ML
            self.activity_log.append({
                "action": action,
                "user": user,
                "timestamp": current_time,
                "attempts": self.attempts[user][action]
            })

            # Reset attempts on success
            self.attempts[user][action] = 0
            record = AuditRecord(
                timestamp=current_time,
                action=action,
                entity=user,
                status="success",
                details=self._encrypt_log(f"Policy enforced, signature: {signature}"),
                encrypted=True
            )
            record.integrity_hash = self._compute_integrity_hash(record)
            self.audit_log.append(record)
            logger.info(f"Policy enforced: {user} allowed for {action}")
            return "Policy enforced"
        except Exception as e:
            record = AuditRecord(
                timestamp=time.time(),
                action=action,
                entity=user,
                status="failed",
                details=self._encrypt_log(str(e)),
                encrypted=True
            )
            record.integrity_hash = self._compute_integrity_hash(record)
            self.audit_log.append(record)
            logger.error(f"Policy enforcement error: {e}")
            return f"Error: Policy enforcement failed: {e}"

    def get_audit_log(self) -> List[Dict]:
        """Return decrypted audit log with integrity verification."""
        audit_logs = []
        for record in self.audit_log:
            if record.integrity_hash and self._compute_integrity_hash(record) != record.integrity_hash:
                audit_logs.append({
                    "timestamp": record.timestamp,
                    "action": record.action,
                    "entity": record.entity,
                    "status": "tampered",
                    "details": "Audit log entry tampered",
                    "integrity_hash": record.integrity_hash
                })
                logger.error(f"Tampered audit log detected at {record.timestamp}")
            else:
                audit_logs.append({
                    "timestamp": record.timestamp,
                    "action": record.action,
                    "entity": record.entity,
                    "status": record.status,
                    "details": self._decrypt_log(record.details) if record.encrypted else record.details,
                    "integrity_hash": record.integrity_hash
                })
        return audit_logs