import uuid

from collections.abc import Callable

from common.client import A2AClient
from common.types import (
    AgentCard,
    Task,
    TaskArtifactUpdateEvent,
    TaskSendParams,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)


TaskCallbackArg = Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
TaskUpdateCallback = Callable[[TaskCallbackArg, AgentCard], Task]


class RemoteAgentConnections:
    """A class to hold the connections to the remote agents."""

    def __init__(self, agent_card: AgentCard):
        self.agent_client = A2AClient(agent_card)
        self.card = agent_card

        self.conversation_name = None
        self.conversation = None
        self.pending_tasks = set()

    def get_agent(self) -> AgentCard:
        return self.card

    async def send_task(
        self,
        request: TaskSendParams,
        task_callback: TaskUpdateCallback | None,
    ) -> Task:
        last_valid_task: Task | None = None

        def ensure_task(obj) -> Task:
            """確保一定回傳 Task"""
            nonlocal last_valid_task

            if isinstance(obj, Task):
                last_valid_task = obj
                return obj

            # TaskStatusUpdateEvent → 包成 Task
            if isinstance(obj, TaskStatusUpdateEvent):
                t = Task(
                    id=request.id,
                    sessionId=request.sessionId,
                    status=obj.status,
                    history=[obj.status.message],
                )
                last_valid_task = t
                return t

            # TaskArtifactUpdateEvent → 也包成 Task
            if isinstance(obj, TaskArtifactUpdateEvent):
                t = Task(
                    id=request.id,
                    sessionId=request.sessionId,
                    status=TaskStatus(state=TaskState.WORKING),
                    artifacts=[obj.artifact],
                )
                last_valid_task = t
                return t

            # callback 回 None → fallback
            if last_valid_task:
                return last_valid_task

            # 最後 fallback：建立一個 Task
            return Task(
                id=request.id,
                sessionId=request.sessionId,
                status=TaskStatus(state=TaskState.WORKING),
                history=[request.message],
            )

        # ─── Streaming mode ──────────────────────────────────────────
        if self.card.capabilities.streaming:

            if task_callback:
                # 初始 SUBMITTED event
                init_task = Task(
                    id=request.id,
                    sessionId=request.sessionId,
                    status=TaskStatus(
                        state=TaskState.SUBMITTED,
                        message=request.message,
                    ),
                    history=[request.message],
                )
                task_callback(init_task, self.card)
                last_valid_task = init_task

            async for response in self.agent_client.send_task_streaming(
                request.model_dump()
            ):
                merge_metadata(response.result, request)

                if task_callback:
                    returned = task_callback(response.result, self.card)
                    ensured = ensure_task(returned)
                    last_valid_task = ensured

                if getattr(response.result, 'final', False):
                    break

            return last_valid_task

        # ─── Non-streaming mode ───────────────────────────────────────
        response = await self.agent_client.send_task(request.model_dump())
        merge_metadata(response.result, request)

        if task_callback:
            returned = task_callback(response.result, self.card)
            ensured = ensure_task(returned)
            last_valid_task = ensured
            return ensured

        # 沒 callback → 直接轉 Task（本來就不會 None）
        return response.result



def merge_metadata(target, source):
    if not hasattr(target, 'metadata') or not hasattr(source, 'metadata'):
        return
    if target.metadata and source.metadata:
        target.metadata.update(source.metadata)
    elif source.metadata:
        target.metadata = dict(**source.metadata)
