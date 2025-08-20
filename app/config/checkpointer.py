
import asyncio
from collections.abc import AsyncIterator
from typing import Any, Optional
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph_checkpoint_aws.saver import BedrockSessionSaver

class FixedBedrockSessionSaver(BedrockSessionSaver):
    """BedrockSessionSaver with async methods implemented"""
    
    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Async version of get_tuple"""
        # 동기 메서드를 executor에서 실행
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_tuple, config)
    
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Async version of put"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.put, config, checkpoint, metadata, new_versions
        )
    
    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: list[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Async version of put_writes"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.put_writes, config, writes, task_id, task_path
        )
    
    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Async version of list"""
        loop = asyncio.get_event_loop()
        
        def _sync_list():
            return list(self.list(config, filter=filter, before=before, limit=limit))
        
        checkpoints = await loop.run_in_executor(None, _sync_list)
        
        for checkpoint in checkpoints:
            yield checkpoint