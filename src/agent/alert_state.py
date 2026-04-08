"""Redis-backed alert state management for patient cooldown tracking."""

from __future__ import annotations

from redis.asyncio import Redis

from src.core.config import Settings


class RedisAlertStateStore:
    """Store per-patient, per-condition alert cooldown state in Redis."""

    def __init__(self, settings: Settings) -> None:
        self._client = Redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        self._key_prefix = settings.redis_alert_key_prefix

    def _key(self, patient_id: str, condition: str) -> str:
        """Build the Redis key for a patient's alert state."""

        return f"{self._key_prefix}:{patient_id}:{condition}"

    async def claim_alert(
        self,
        patient_id: str,
        condition: str,
        cooldown_seconds: int,
        value: str,
    ) -> tuple[bool, int]:
        """Atomically claim a cooldown slot for a patient alert.

        Returns:
            tuple[bool, int]: Whether the alert slot was acquired, and the remaining
            cooldown in seconds when the claim failed.
        """

        key = self._key(patient_id, condition)
        for _ in range(2):
            acquired = await self._client.set(key, value, ex=cooldown_seconds, nx=True)
            if acquired:
                return True, 0

            remaining_cooldown_seconds = await self._client.ttl(key)
            if remaining_cooldown_seconds is not None and remaining_cooldown_seconds > 0:
                return False, remaining_cooldown_seconds

        return False, 0

    async def clear_alert(self, patient_id: str, condition: str) -> None:
        """Clear any existing cooldown state for a patient-condition pair."""

        await self._client.delete(self._key(patient_id, condition))

    async def clear_patient_alerts(self, patient_id: str) -> None:
        """Clear all cooldown states for a patient."""

        pattern = f"{self._key_prefix}:{patient_id}:*"
        cursor = 0
        keys_to_delete: list[str] = []

        while True:
            cursor, keys = await self._client.scan(cursor=cursor, match=pattern, count=50)
            keys_to_delete.extend(keys)
            if cursor == 0:
                break

        if keys_to_delete:
            await self._client.delete(*keys_to_delete)

    async def close(self) -> None:
        """Close the Redis client gracefully."""

        await self._client.aclose()
