import asyncio
import contextlib
import multiprocessing
import queue
from collections.abc import AsyncIterator

from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect


class PreviewBroadcaster:
    """
    Forwards the runner's previews to the WebSockets watching each job.
    """

    def __init__(
        self,
    ):
        # Lauryn spawns its GPU workers, and a queue can only be shared with processes from its own context
        self.queue = multiprocessing.get_context("spawn").Queue()
        self.subscribers: dict[str, set[WebSocket]] = {}
        self.router = APIRouter(lifespan=self.lifespan)
        self.router.add_api_websocket_route("/preview/{preview_id}", self.subscribe)

    @contextlib.asynccontextmanager
    async def lifespan(
        self,
        app: FastAPI,
    ) -> AsyncIterator[None]:
        """
        Forward previews while the app runs.

        :param app: The app `router` is included in.
        :type app: FastAPI
        """

        forwarder = asyncio.create_task(self.forward())
        yield
        forwarder.cancel()

    async def forward(
        self,
    ):
        """
        Send each preview to the WebSockets watching its job, dropping all but the latest of a backlog.
        """

        while True:
            try:
                preview_id, preview = await asyncio.to_thread(self.queue.get, timeout=1.0)
            except queue.Empty:
                continue
            latest = {preview_id: preview}
            while True:
                try:
                    preview_id, preview = self.queue.get_nowait()
                except queue.Empty:
                    break
                latest[preview_id] = preview

            for preview_id, preview in latest.items():
                for websocket in list(self.subscribers.get(preview_id, ())):
                    with contextlib.suppress(WebSocketDisconnect, RuntimeError):  # `subscribe()` removes it
                        await websocket.send_bytes(preview)

    async def subscribe(
        self,
        websocket: WebSocket,
        preview_id: str,
    ):
        """
        Accept a WebSocket and send it the job's previews until it disconnects.

        :param websocket: The client's WebSocket.
        :type websocket: WebSocket
        :param preview_id: The `previewId` returned when the job was submitted.
        :type preview_id: str
        """

        await websocket.accept()
        self.subscribers.setdefault(preview_id, set()).add(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            subscribers = self.subscribers[preview_id]
            subscribers.discard(websocket)
            if not subscribers:
                del self.subscribers[preview_id]
