# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from datetime import UTC, datetime
from threading import RLock
from uuid import uuid4

from overtourism.backend.auth.identity.users import User, UserRepository, UserRole


class UserManager:
    def __init__(self, repository: UserRepository) -> None:
        self._repository = repository
        self._lock = RLock()
        self._users_by_id: dict[str, User] = {}
        self._users_by_subject: dict[str, User] = {}
        self.reload()

    def reload(self) -> None:
        with self._lock:
            self._reload_locked()

    def list_users(self, territory: str | None = None) -> list[User]:
        with self._lock:
            users = list(self._users_by_id.values())
            if territory is None:
                return users
            return [user for user in users if territory in user.territories]

    def get_active_user_by_subject(self, subject: str) -> User | None:
        normalized_subject = subject.strip()
        with self._lock:
            user = self._users_by_subject.get(normalized_subject)
            return user if user is not None and user.is_active else None

    def claim_user_by_identifier(self, identifier: str, subject: str) -> User | None:
        normalized_identifier = identifier.strip().casefold()
        normalized_subject = subject.strip()
        if not normalized_identifier or not normalized_subject:
            return None

        with self._lock:
            self._repository.bind_subject_if_unlinked(
                normalized_identifier,
                normalized_subject,
            )
            self._reload_locked()
            user = self._users_by_subject.get(normalized_subject)
            return user if user is not None and user.is_active else None

    def create_user(
        self,
        *,
        identifier: str,
        role: UserRole,
        territories: Iterable[str],
    ) -> User:
        normalized_identifier = identifier.strip().casefold()
        if not normalized_identifier:
            raise ValueError("User identifier must not be empty")

        normalized_role = UserRole(role)
        normalized_territories = self._normalize_territories(territories)
        self._validate_territories(normalized_role, normalized_territories)
        now = datetime.now(UTC)
        user = User(
            user_id=uuid4().hex,
            identifier=normalized_identifier,
            subject=None,
            role=normalized_role,
            is_active=True,
            territories=normalized_territories,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._repository.save_user(user)
            self._reload_locked()
            return self._users_by_id[user.user_id]

    def create_first_admin(self, identifier: str) -> User:
        normalized_identifier = identifier.strip().casefold()
        if not normalized_identifier:
            raise ValueError("User identifier must not be empty")

        now = datetime.now(UTC)
        user = User(
            user_id=uuid4().hex,
            identifier=normalized_identifier,
            subject=None,
            role=UserRole.ADMIN,
            is_active=True,
            territories=frozenset(),
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            if not self._repository.create_first_admin_if_empty(user):
                self._reload_locked()
                raise ValueError("The user registry is not empty")
            self._reload_locked()
            return self._users_by_id[user.user_id]

    def update_user(
        self,
        user_id: str,
        *,
        role: UserRole | None = None,
        territories: Iterable[str] | None = None,
        is_active: bool | None = None,
    ) -> User:
        with self._lock:
            current_user = self._users_by_id.get(user_id)
            if current_user is None:
                raise KeyError(f"User '{user_id}' not found")

            updated_role = current_user.role if role is None else UserRole(role)
            if territories is None:
                updated_territories = (
                    frozenset()
                    if updated_role is UserRole.ADMIN
                    else current_user.territories
                )
            else:
                updated_territories = self._normalize_territories(territories)
            self._validate_territories(updated_role, updated_territories)

            updated_user = replace(
                current_user,
                role=updated_role,
                territories=updated_territories,
                is_active=(current_user.is_active if is_active is None else is_active),
                updated_at=datetime.now(UTC),
            )
            self._repository.save_user(updated_user)
            self._reload_locked()
            return self._users_by_id[user_id]

    def deactivate_user(self, user_id: str) -> User:
        return self.update_user(user_id, is_active=False)

    def _reload_locked(self) -> None:
        try:
            users = self._repository.load_users()
            users_by_id = {user.user_id: user for user in users}
            users_by_subject = {
                user.subject: user for user in users if user.subject is not None
            }
        except Exception:
            self._users_by_id = {}
            self._users_by_subject = {}
            raise

        self._users_by_id = users_by_id
        self._users_by_subject = users_by_subject

    def _normalize_territories(
        self,
        territories: Iterable[str],
    ) -> frozenset[str]:
        normalized = (territory.strip() for territory in territories)
        return frozenset(territory for territory in normalized if territory)

    def _validate_territories(
        self,
        role: UserRole,
        territories: frozenset[str],
    ) -> None:
        if role is UserRole.ADMIN and territories:
            raise ValueError("Admin users must not have territory assignments")
        if role is UserRole.MULTIEDITOR and not territories:
            raise ValueError("Multieditor users require at least one territory")
