"""Tests that retried training writes execute only once."""

import asyncio

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.tinker import types
from skyrl.tinker.api import (
    SaveWeightsRequest, SaveWeightsForSamplerRequest, create_future,
    save_weights, save_weights_for_sampler,
)
from skyrl.tinker.db_models import (
    CheckpointDB, FutureDB, ModelDB, SamplingSessionDB, SessionDB, enable_sqlite_wal,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("sampler", [False, True])
async def test_checkpoint_retry_reuses_future_and_rejects_conflicts(tmp_path, sampler):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'checkpoint.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine) as session:
        session.add(SessionDB(session_id="session_1", sdk_version="test"))
        await session.commit()
        session.add(ModelDB(model_id="model_1", base_model="test", lora_config={},
                            status="ready", request_id=1, session_id="session_1"))
        await session.commit()

    endpoint = save_weights_for_sampler if sampler else save_weights
    request_cls = SaveWeightsForSamplerRequest if sampler else SaveWeightsRequest
    extra = {"sampling_session_seq_id": 3} if sampler else {}
    request = request_cls(model_id="model_1", path="000027", seq_id=7, **extra)
    async with AsyncSession(engine) as session:
        original = await endpoint(request, session)
    # Use a fresh, expiring session, just as a new HTTP request would.
    async with AsyncSession(engine) as session:
        retry = await endpoint(request, session)
        assert retry.future_id == original.future_id
        assert len((await session.exec(select(FutureDB))).all()) == 1
        assert len((await session.exec(select(CheckpointDB))).all()) == 1
        assert len((await session.exec(select(SamplingSessionDB))).all()) == int(sampler)
    for path, seq_id in [("different", 7), ("000027", 8)]:
        async with AsyncSession(engine) as session:
            with pytest.raises(HTTPException) as error:
                await endpoint(request_cls(model_id="model_1", path=path, seq_id=seq_id, **extra), session)
            assert error.value.status_code == 409
    await engine.dispose()


def optim_input(learning_rate: float = 1e-4) -> types.OptimStepInput:
    return types.OptimStepInput(
        adam_params=types.AdamParams(
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.95,
            eps=1e-12,
            weight_decay=0.0,
        )
    )


@pytest.mark.asyncio
async def test_training_request_retry_returns_original_future(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tinker.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)

    async with AsyncSession(engine) as session:
        original_id = await create_future(session, types.RequestType.OPTIM_STEP, "model_1", optim_input(), seq_id=7)
        await session.commit()

    async with AsyncSession(engine) as session:
        retry_id = await create_future(session, types.RequestType.OPTIM_STEP, "model_1", optim_input(), seq_id=7)
        await session.commit()
        futures = (await session.exec(select(FutureDB))).all()

    assert retry_id == original_id
    assert len(futures) == 1
    assert futures[0].seq_id == 7
    await engine.dispose()


@pytest.mark.asyncio
async def test_training_request_sequence_reuse_with_new_payload_fails(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tinker.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)

    async with AsyncSession(engine) as session:
        await create_future(session, types.RequestType.OPTIM_STEP, "model_1", optim_input(), seq_id=7)
        await session.commit()

    async with AsyncSession(engine) as session:
        with pytest.raises(HTTPException, match="sequence number was reused") as error:
            await create_future(session, types.RequestType.OPTIM_STEP, "model_1", optim_input(2e-4), seq_id=7)

    assert error.value.status_code == 409
    await engine.dispose()


@pytest.mark.asyncio
async def test_concurrent_training_request_types_cannot_reuse_sequence(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tinker.db'}")
    enable_sqlite_wal(engine.sync_engine)
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)

    start = asyncio.Event()

    async def create(request_type, request_data):
        async with AsyncSession(engine) as session:
            await start.wait()
            try:
                request_id = await create_future(session, request_type, "model_1", request_data, seq_id=7)
                await session.commit()
                return request_id
            except HTTPException as error:
                return error

    requests = [
        asyncio.create_task(
            create(
                types.RequestType.FORWARD_BACKWARD,
                types.ForwardBackwardInput(data=[], loss_fn="cross_entropy"),
            )
        ),
        asyncio.create_task(create(types.RequestType.OPTIM_STEP, optim_input())),
    ]
    start.set()
    results = await asyncio.gather(*requests)

    successes = [result for result in results if isinstance(result, int)]
    conflicts = [result for result in results if isinstance(result, HTTPException)]
    assert len(successes) == 1
    assert len(conflicts) == 1
    assert conflicts[0].status_code == 409
    assert "sequence number was reused" in conflicts[0].detail

    async with AsyncSession(engine) as session:
        futures = (await session.exec(select(FutureDB))).all()
    assert len(futures) == 1
    assert futures[0].request_id == successes[0]
    assert futures[0].seq_id == 7
    await engine.dispose()


@pytest.mark.asyncio
async def test_training_requests_without_sequence_numbers_remain_distinct(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tinker.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)

    async with AsyncSession(engine) as session:
        first_id = await create_future(session, types.RequestType.OPTIM_STEP, "model_1", optim_input())
        second_id = await create_future(session, types.RequestType.OPTIM_STEP, "model_1", optim_input())
        await session.commit()
        futures = (await session.exec(select(FutureDB))).all()

    # Two identical requests for one model, both with seq_id NULL, must not collide on
    # the futures (model_id, seq_id) constraint: SQL counts NULLs as distinct. That is
    # what keeps clients which send no seq_id on the pre-idempotency behaviour.
    assert second_id != first_id
    assert len(futures) == 2
    assert [future.seq_id for future in futures] == [None, None]
    await engine.dispose()
