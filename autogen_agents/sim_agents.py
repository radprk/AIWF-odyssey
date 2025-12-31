from datetime import datetime
from memory import log_interaction

class TaskAwareAgent:
    """Wraps a raw AutoGen AssistantAgent with bookkeeping for cost & duration
    plus a tiny bit of logging.  This is the one referenced by base_agents.py."""

    def __init__(self, agent, role: str, hourly_wage: float):
        self.agent = agent
        self.role = role
        self.hourly_wage = hourly_wage

    # ---------------------------------------------------------------------
    # Public helpers -------------------------------------------------------
    # ---------------------------------------------------------------------
    def handle(
        self,
        query: str,
        customer_proxy,
        task_type: str,
        duration_est_sec: int,
        context: str | None = None,
        metadata: dict | None = None,
    ):
        """Send *query* to the wrapped LLM assistant.  We:
        1. Print a small banner so you can watch the sim run.
        2. Time the round‑trip to produce *actual* duration.
        3. Compute a dollar cost based on *actual* duration × wage.
        4. Persist the transcript header (query / summary) via memory.log_interaction.
        """

        print(f"🛠️ {self.role} handling task: {task_type} (est {duration_est_sec}s)")

        start_time = datetime.now()
        # We prefix the user query with context only visible to the assistant
        # (makes the prompt a bit richer & lets us see the ETA up front)
        prompt_parts = [f"[{task_type} – est {duration_est_sec}s]"]
        if context:
            prompt_parts.append(f"\n[Retrieved Context]\n{context}")
        prompt_parts.append(f"\n{query}")
        prompt = "\n".join(prompt_parts)
        result = customer_proxy.initiate_chat(
            message=prompt,
            recipient=self.agent,
            summary_method="last_msg",
            max_turns=2,
        )
        end_time = datetime.now()

        actual_sec = (end_time - start_time).total_seconds()
        cost = round(actual_sec / 3600 * self.hourly_wage, 2)
        print(f"✅ {self.role} completed in {actual_sec:.1f}s – cost ${cost:.2f}")

        # -----------------------------------------------------------------
        # Persist lightweight trace (for dashboards later) -----------------
        # -----------------------------------------------------------------
        merged_metadata = dict(metadata or {})
        merged_metadata.update({
            "role": self.role,
            "actual_sec": round(actual_sec, 2),
            "cost": cost,
        })
        log_interaction(
            agent_name=self.agent.name,
            query=query,
            response=result.summary,
            escalated_to=None,
            confidence=None,
            metadata=merged_metadata,
        )

        return result.summary


class TaskAwareGroupChat:
    """Optional convenience wrapper for group‑chat simulations."""

    def __init__(self, manager, role: str, hourly_wage: float):
        self.manager = manager
        self.role = role
        self.hourly_wage = hourly_wage

    def handle(
        self,
        query: str,
        customer_proxy,
        task_type: str,
        duration_est_sec: int,
        context: str | None = None,
        metadata: dict | None = None,
    ):
        print(f"🤝 GroupChat handling task: {task_type} (est {duration_est_sec}s)")

        start_time = datetime.now()
        prompt_parts = [f"[{task_type} – est {duration_est_sec}s]"]
        if context:
            prompt_parts.append(f"\n[Retrieved Context]\n{context}")
        prompt_parts.append(f"\n{query}")
        result = customer_proxy.initiate_chat(
            self.manager,
            message="\n".join(prompt_parts),
        )
        end_time = datetime.now()

        actual_sec = (end_time - start_time).total_seconds()
        cost = round(actual_sec / 3600 * self.hourly_wage, 2)
        print(f"💬 GroupChat completed in {actual_sec:.1f}s – cost ${cost:.2f}")

        merged_metadata = dict(metadata or {})
        merged_metadata.update({
            "role": self.role,
            "actual_sec": round(actual_sec, 2),
            "cost": cost,
        })
        log_interaction(
            agent_name="GroupChatManager",
            query=query,
            response=result.summary,
            escalated_to=None,
            confidence=None,
            metadata=merged_metadata,
        )

        return result.summary
