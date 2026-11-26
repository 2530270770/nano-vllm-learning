
from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def make_scheduler(
    max_num_seqs=8,
    max_num_decode_seqs=8,
    max_num_batched_tokens=1024,
    num_kvcache_blocks=32,
    max_num_prefill_seqs=None,
    enable_prefill_aging=False,
):
    """创建不依赖真实模型和 GPU 的 Scheduler。"""
    config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_decode_seqs=max_num_decode_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        eos=-1,
        kvcache_block_size=256,
        num_kvcache_blocks=num_kvcache_blocks,
        max_num_prefill_seqs=max_num_prefill_seqs,
        enable_prefill_aging=enable_prefill_aging,
    )
    return Scheduler(config)


def make_seq(num_tokens, max_tokens=16):
    """创建指定长度的 Sequence。"""
    return Sequence(
        token_ids=list(range(num_tokens)),
        sampling_params=SamplingParams(
            max_tokens=max_tokens,
            ignore_eos=True,
        ),
    )


def add_running_seq(scheduler, seq):
    """将 Sequence 初始化为已经完成 Prefill 的 Decode 请求。"""
    scheduler.block_manager.allocate(seq, 0)
    seq.num_cached_tokens = seq.num_tokens
    seq.status = SequenceStatus.RUNNING
    seq.is_prefill = False
    scheduler.running.append(seq)


def test_only_prefill():
    """只有 Waiting 请求时，应该调度 Prefill。"""
    scheduler = make_scheduler()
    seq = make_seq(20)
    scheduler.add(seq)

    output = scheduler.schedule()

    assert output.decode_seqs == []
    assert output.prefill_seqs == [seq]
    assert output.decode_tokens == 0
    assert output.prefill_tokens == 20
    assert seq.status == SequenceStatus.RUNNING


def test_only_decode():
    """只有 Running 请求时，应该调度 Decode。"""
    scheduler = make_scheduler()
    seq = make_seq(20)
    add_running_seq(scheduler, seq)

    output = scheduler.schedule()

    assert output.decode_seqs == [seq]
    assert output.prefill_seqs == []
    assert output.decode_tokens == 1
    assert output.prefill_tokens == 0
    assert seq.num_scheduled_tokens == 1
    assert seq.is_prefill is False


def test_mixed_decode_and_prefill():
    """同一轮应该可以同时调度 Decode 和 Prefill。"""
    scheduler = make_scheduler(
        max_num_seqs=4,
        max_num_decode_seqs=2,
        max_num_batched_tokens=128,
    )

    decode_seq = make_seq(20)
    prefill_seq = make_seq(30)

    add_running_seq(scheduler, decode_seq)
    scheduler.add(prefill_seq)

    output = scheduler.schedule()

    assert output.decode_seqs == [decode_seq]
    assert output.prefill_seqs == [prefill_seq]
    assert output.decode_tokens == 1
    assert output.prefill_tokens == 30

    assert decode_seq.status == SequenceStatus.RUNNING
    assert prefill_seq.status == SequenceStatus.RUNNING


def test_decode_respects_shared_budget_and_independent_limit():
    """Decode 数量不能超过独立上限、请求数预算和 Token 预算。"""
    scheduler = make_scheduler(
        max_num_seqs=10,
        max_num_decode_seqs=8,
        max_num_batched_tokens=3,
    )

    seqs = [make_seq(20) for _ in range(5)]

    for seq in seqs:
        add_running_seq(scheduler, seq)

    output = scheduler.schedule()

    assert len(output.decode_seqs) == 3
    assert output.decode_tokens == 3
    assert output.prefill_seqs == []


def test_kv_allocation_failure_skips_to_next_request():
    """前一个请求 KV Cache 不足时，不能阻塞后续请求。"""
    scheduler = make_scheduler(
        max_num_seqs=4,
        max_num_batched_tokens=100,
        num_kvcache_blocks=1,
    )

    # 300 个 Token 需要两个 KV Cache Block，当前只有一个 Block。
    large_seq = make_seq(300)

    # 10 个 Token 只需要一个 Block。
    small_seq = make_seq(10)

    scheduler.add(large_seq)
    scheduler.add(small_seq)

    output = scheduler.schedule()

    assert output.prefill_seqs == [small_seq]
    assert output.prefill_tokens == 10

    # 大请求没有被丢弃，仍然留在 Waiting 队列中。
    assert large_seq in scheduler.waiting

    # 小请求成功获得 KV Cache。
    assert len(small_seq.block_table) == 1


def test_integer_water_filling():
    """验证多个 Prefill 请求之间的整数 Token 分配。"""
    scheduler = make_scheduler(
        max_num_seqs=3,
        max_num_batched_tokens=100,
        num_kvcache_blocks=3,
    )

    seqs = [
        make_seq(40),
        make_seq(40),
        make_seq(100),
    ]

    for seq in seqs:
        scheduler.add(seq)

    output = scheduler.schedule()

    allocations = [
        seq.num_scheduled_tokens
        for seq in output.prefill_seqs
    ]

    # 水位为 33，剩余的一个 Token 分配给第一个未满足需求的请求。
    assert allocations == [34, 33, 33]
    assert sum(allocations) == 100
    assert output.prefill_tokens == 100


def test_chunk_prefill_across_multiple_steps():
    """一个长 Prefill 请求应该可以跨多个 Step 完成。"""
    scheduler = make_scheduler(
        max_num_seqs=1,
        max_num_batched_tokens=100,
        num_kvcache_blocks=2,
    )

    seq = make_seq(300)
    scheduler.add(seq)

    # 第一个 Step：处理 100 个 Token。
    output1 = scheduler.schedule()

    assert output1.prefill_seqs == [seq]
    assert seq.num_scheduled_tokens == 100
    assert seq.status == SequenceStatus.WAITING

    scheduler.postprocess([seq], [999], is_prefill=True)

    assert seq.num_cached_tokens == 100
    assert seq.status == SequenceStatus.WAITING

    # 第二个 Step：再处理 100 个 Token。
    output2 = scheduler.schedule()

    assert output2.prefill_seqs == [seq]
    assert seq.num_scheduled_tokens == 100

    scheduler.postprocess([seq], [999], is_prefill=True)

    assert seq.num_cached_tokens == 200
    assert seq.status == SequenceStatus.WAITING

    # 第三个 Step：处理最后 100 个 Token。
    output3 = scheduler.schedule()

    assert output3.prefill_seqs == [seq]
    assert seq.num_scheduled_tokens == 100

    scheduler.postprocess([seq], [999], is_prefill=True)

    assert seq.num_cached_tokens == 300
    assert seq.status == SequenceStatus.RUNNING

    # 完成 Prefill 的请求只能从下一轮开始进入 Decode。
    output4 = scheduler.schedule()

    assert output4.decode_seqs == [seq]
    assert output4.prefill_seqs == []

def test_decode_respects_independent_limit():
    """Decode 数量不能超过 max_num_decode_seqs。"""
    scheduler = make_scheduler(
        max_num_seqs=10,
        max_num_decode_seqs=2,
        max_num_batched_tokens=100,
    )

    seqs = [make_seq(20) for _ in range(5)]

    for seq in seqs:
        add_running_seq(scheduler, seq)

    output = scheduler.schedule()

    assert len(output.decode_seqs) == 2
    assert output.decode_tokens == 2

    # 其余请求仍然留在 Running 队列中。
    assert len(scheduler.running) == 5




def test_decode_preempts_another_request_when_kv_cache_is_full():
    """KV Cache 不足时，优先抢占尚未调度的 Decode 请求。"""
    scheduler = make_scheduler(
        max_num_seqs=4,
        max_num_decode_seqs=4,
        max_num_batched_tokens=100,
        num_kvcache_blocks=2,
    )

    seq1 = make_seq(256)
    seq2 = make_seq(256)

    add_running_seq(scheduler, seq1)
    add_running_seq(scheduler, seq2)

    # 模拟两个请求都已经生成了一个 Token，
    # 当前长度变为 257，下一个 Decode 需要新增一个 KV Block。
    seq1.append_token(999)
    seq2.append_token(999)

    output = scheduler.schedule()

    # 第一个请求获得新增 Block，进入 Decode Batch。
    assert output.decode_seqs == [seq1]

    # 第二个请求被抢占，回到 Waiting。
    assert seq2.status == SequenceStatus.WAITING
    assert seq2 in scheduler.waiting
    assert seq2.block_table == []

    # 第一个请求现在拥有两个 Block。
    assert len(seq1.block_table) == 2

def test_decode_preempts_itself_when_no_other_request_can_be_preempted():
    """只有一个 Decode 请求且 KV Cache 不足时，抢占自身。"""
    scheduler = make_scheduler(
        max_num_seqs=4,
        max_num_decode_seqs=4,
        max_num_batched_tokens=100,
        num_kvcache_blocks=1,
    )

    seq = make_seq(256)
    add_running_seq(scheduler, seq)

    # 模拟已经生成一个 Token，需要新增 KV Block。
    seq.append_token(999)

    output = scheduler.schedule()

    # 没有可用 Block，也没有其他请求可以抢占。
    assert output.decode_seqs == []

    # 当前请求被抢占并放回 Waiting。
    assert seq.status == SequenceStatus.WAITING
    assert seq in scheduler.waiting

    # 原来的 KV Block 已经释放。
    assert seq.block_table == []
    assert len(scheduler.block_manager.free_block_ids) == 1

def test_decode_preemption_keeps_block_accounting_consistent():
    """Decode 抢占后，Block 的使用情况和引用计数必须一致。"""
    scheduler = make_scheduler(
        max_num_seqs=4,
        max_num_decode_seqs=4,
        max_num_batched_tokens=100,
        num_kvcache_blocks=2,
    )

    seq1 = make_seq(256)
    seq2 = make_seq(256)

    add_running_seq(scheduler, seq1)
    add_running_seq(scheduler, seq2)

    seq1.append_token(999)
    seq2.append_token(999)

    output = scheduler.schedule()

    assert output.decode_seqs == [seq1]

    # seq1 占用两个 Block。
    assert len(seq1.block_table) == 2

    # seq2 被抢占，释放了自己的 Block。
    assert seq2.block_table == []
    assert seq2.status == SequenceStatus.WAITING

    block_manager = scheduler.block_manager

    # 两个 Block 都被 seq1 使用。
    assert len(block_manager.used_block_ids) == 2
    assert len(block_manager.free_block_ids) == 0

    # 两个 Block 的引用计数都应该是 1。
    for block_id in seq1.block_table:
        assert block_manager.blocks[block_id].ref_count == 1

    # 每个 Block 只能处于 Used 或 Free 其中一种状态。
    assert (
        block_manager.used_block_ids
        .isdisjoint(set(block_manager.free_block_ids))
    )

    assert (
        len(block_manager.used_block_ids)
        + len(block_manager.free_block_ids)
        == len(block_manager.blocks)
    )




def test_prefill_takes_decode_slot_after_decode_finishes():
    """Decode 请求结束后，已完成 Prefill 的请求应该接管 Decode 名额。"""
    scheduler = make_scheduler(
        max_num_seqs=2,
        max_num_decode_seqs=1,
        max_num_batched_tokens=4,
    )

    # 已经完成 Prefill 的 Decode 请求，只允许再生成一个 Token。
    decode_seq = make_seq(10, max_tokens=1)
    add_running_seq(scheduler, decode_seq)

    # 新到达的 Prefill 请求。
    prefill_seq = make_seq(2, max_tokens=4)
    scheduler.add(prefill_seq)

    # Step 1：Decode 占用唯一 Decode 名额，
    # Prefill 利用剩余预算完成计算。
    output1 = scheduler.schedule()

    assert output1.decode_seqs == [decode_seq]
    assert output1.prefill_seqs == [prefill_seq]
    assert output1.decode_tokens == 1
    assert output1.prefill_tokens == 2

    # 分别执行两个阶段的 Postprocess。
    scheduler.postprocess(
        output1.decode_seqs,
        [999],
        is_prefill=False,
    )
    scheduler.postprocess(
        output1.prefill_seqs,
        [888],
        is_prefill=True,
    )

    # 原 Decode 请求已经结束。
    assert decode_seq.status == SequenceStatus.FINISHED

    # Prefill 请求已经完成 Prefill，进入 Running。
    assert prefill_seq.status == SequenceStatus.RUNNING
    assert prefill_seq.num_completion_tokens == 1

    # Step 2：原 Decode 请求退出后，
    # Prefill 请求应该接管唯一的 Decode 名额。
    output2 = scheduler.schedule()

    assert output2.decode_seqs == [prefill_seq]
    assert output2.prefill_seqs == []
    assert output2.decode_tokens == 1
    assert output2.prefill_tokens == 0


def test_prefill_respects_remaining_slots_and_keeps_waiting_requests():
    """Prefill 名额不足时，只调度允许数量，其余请求不能丢失。"""
    scheduler = make_scheduler(
        max_num_seqs=2,
        max_num_decode_seqs=1,
        max_num_batched_tokens=10,
        num_kvcache_blocks=8,
    )

    # 一个 Decode 请求占用一个请求名额和一个 Token Budget。
    decode_seq = make_seq(20)
    add_running_seq(scheduler, decode_seq)

    # 还有三个等待 Prefill 的请求。
    prefill_seqs = [
        make_seq(20),
        make_seq(20),
        make_seq(20),
    ]

    for seq in prefill_seqs:
        scheduler.add(seq)

    output = scheduler.schedule()

    # Decode 占了一个名额，Prefill 只能使用剩余的一个名额。
    assert output.decode_seqs == [decode_seq]
    assert len(output.prefill_seqs) == 1
    assert output.prefill_seqs == [prefill_seqs[0]]

    # Decode 使用 1 个 Token，Prefill 使用剩余的 9 个。
    assert output.decode_tokens == 1
    assert output.prefill_tokens == 9
    assert prefill_seqs[0].num_scheduled_tokens == 9

    # 第一个 Prefill 被切成 Chunk，回到 Waiting 队尾。
    # 另外两个请求仍在 Waiting 中，且没有丢失。
    assert list(scheduler.waiting) == [
        prefill_seqs[1],
        prefill_seqs[2],
        prefill_seqs[0],
    ]

    assert all(
        seq.status == SequenceStatus.WAITING
        for seq in prefill_seqs
    )

def test_prefill_respects_concurrency_limit():
    """Prefill 每轮只能调度指定数量的请求。"""
    scheduler = make_scheduler(
        max_num_seqs=8,
        max_num_batched_tokens=100,
        num_kvcache_blocks=8,
        max_num_prefill_seqs=2,
    )

    seqs = [make_seq(20) for _ in range(4)]

    for seq in seqs:
        scheduler.add(seq)

    output = scheduler.schedule()

    # 本轮最多调度两个 Prefill 请求。
    assert output.prefill_seqs == seqs[:2]
    assert len(output.prefill_seqs) == 2

    # 两个请求都完成了 Prefill。
    assert all(
        seq.status == SequenceStatus.RUNNING
        for seq in seqs[:2]
    )

    # 另外两个请求仍然在 Waiting 队列中。
    assert list(scheduler.waiting) == seqs[2:]


def test_prefill_concurrency_limit_with_decode():
    """Prefill 并发上限不能影响 Decode 优先级。"""
    scheduler = make_scheduler(
        max_num_seqs=4,
        max_num_decode_seqs=1,
        max_num_batched_tokens=64,
        num_kvcache_blocks=8,
        max_num_prefill_seqs=1,
    )

    decode_seq = make_seq(20)
    add_running_seq(scheduler, decode_seq)

    prefill_seqs = [
        make_seq(20),
        make_seq(20),
        make_seq(20),
    ]

    for seq in prefill_seqs:
        scheduler.add(seq)

    output = scheduler.schedule()

    # Decode 仍然优先。
    assert output.decode_seqs == [decode_seq]
    assert output.decode_tokens == 1

    # Prefill 只能调度一个请求。
    assert output.prefill_seqs == [prefill_seqs[0]]
    assert output.prefill_tokens == 20

    # 其余 Prefill 请求仍然保留在 Waiting 队列。
    assert list(scheduler.waiting) == prefill_seqs[1:]

def test_prefill_aging_prioritizes_older_request():
    """等待时间更长的 Prefill 请求应该优先获得调度名额。"""
    scheduler = make_scheduler(
        max_num_seqs=4,
        max_num_batched_tokens=100,
        num_kvcache_blocks=4,
        max_num_prefill_seqs=1,
        enable_prefill_aging=True,
    )

    newer_seq = make_seq(20)
    older_seq = make_seq(20)

    # 新请求先进入队列，老请求后进入队列。
    scheduler.add(newer_seq)
    scheduler.add(older_seq)

    # 模拟老请求已经等待了更多轮。
    older_seq.prefill_wait_steps = 5

    output = scheduler.schedule()

    # 虽然 newer_seq 在队列前面，
    # 但 older_seq 等待时间更长，应该先被调度。
    assert output.prefill_seqs == [older_seq]

    assert older_seq.prefill_wait_steps == 0

    # 新请求没有被丢弃。
    assert list(scheduler.waiting) == [newer_seq]

    # 未获得调度名额的新请求年龄继续增加。
    assert newer_seq.prefill_wait_steps == 1


def test_prefill_aging_disabled_preserves_fifo():
    """关闭 Aging 时，保持原有 FIFO 候选选择行为。"""
    scheduler = make_scheduler(
        max_num_seqs=4,
        max_num_batched_tokens=100,
        num_kvcache_blocks=4,
        max_num_prefill_seqs=1,
        enable_prefill_aging=False,
    )

    newer_seq = make_seq(20)
    older_seq = make_seq(20)

    scheduler.add(newer_seq)
    scheduler.add(older_seq)

    older_seq.prefill_wait_steps = 5

    output = scheduler.schedule()

    # 关闭 Aging 时仍然按照原始队列顺序。
    assert output.prefill_seqs == [newer_seq]

    # 关闭 Aging 不会修改等待计数。
    assert older_seq.prefill_wait_steps == 5

    assert list(scheduler.waiting) == [older_seq]