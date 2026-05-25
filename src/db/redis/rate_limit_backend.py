"""src/db/redis/rate_limit_backend.py — H6 Redis 限流后端(DEV_SPEC §5.3 + §8.4 H6)。

实现 G3 `RateLimitBackend` Protocol(`is_allowed(key, limit, window) -> (allowed, retry_after)`),
让多实例部署共享配额。算法:**Redis ZSET 滑动窗口**(与 G3 内存版语义完全一致)。

为什么不用 spec §8.4 H6 提示的"SCRIPT INCR + EXPIRE":
- INCR + EXPIRE 是**固定窗口**算法(每整点重置),整点会有 2 倍配额突刺
- G3 内存版钦定**滑动窗口**(每请求 evict 过期戳),Redis 后端必须维持同语义,
  否则线上切到 Redis 后限流行为会跳变,运维不可控
- ZSET 实现的滑动窗口性能开销 ~5 个 Redis 命令/请求,对热路径 30 QPS 的限流
  系统完全可承受

算法(用 Lua 脚本保证原子,避免 race):
  1. ZREMRANGEBYSCORE   key, -inf, now-window   (清窗外戳)
  2. ZCARD              key                     (当前窗口请求数)
  3. 若 < limit:ZADD key now now;EXPIRE key window;返回 (1, 0)
  4. 若 >= limit:ZRANGE key 0 0 WITHSCORES;计算 retry_after;返回 (0, retry_after)

Redis 不可用时降级:**fail-open**(放行)— 限流是 best-effort 防滥用,Redis 挂掉
不应整站 503。WARNING 日志触发告警(spec §5.2.2 表"Redis 连接失败 连续 3 次健康
检查失败 → Warning")。
"""
from __future__ import annotations

import logging
import time

from redis.exceptions import RedisError

from src.common.metrics import redis_command_latency_seconds
from src.db.redis.cache import get_redis_client


_logger = logging.getLogger(__name__)


# Lua 脚本:5 步原子化(spec §8.4 H6 要求 INCR + EXPIRE 原子性,这里是滑动窗口的对应)
# KEYS[1] = bucket key
# ARGV    = now_micro, window_micro, limit
# return  = {allowed (0/1), retry_after_seconds_int}
_SLIDING_WINDOW_LUA = """
local key      = KEYS[1]
local now      = tonumber(ARGV[1])
local window   = tonumber(ARGV[2])
local limit    = tonumber(ARGV[3])
local cutoff   = now - window

redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff)
local count = redis.call('ZCARD', key)

if count < limit then
    redis.call('ZADD', key, now, now)
    redis.call('PEXPIRE', key, math.floor(window / 1000))
    return {1, 0}
else
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local retry  = math.floor((tonumber(oldest[2]) + window - now) / 1000000) + 1
    if retry < 1 then retry = 1 end
    return {0, retry}
end
"""


class RedisSlidingWindow:
    """ZSET 滑动窗口(与 G3 InMemorySlidingWindow 语义对等,可热替换)。

    `key` 命名空间 `ratelimit:<原 key>`(`原 key` 来自 `_key_for_request`,
    形式 `user:<sub>` 或 `ip:<addr>`)。

    Redis 不可用 → fail-open(放行,不抛)。
    """

    def __init__(self) -> None:
        self._script_sha: str | None = None  # SCRIPT LOAD 后缓存 sha,避免每次传脚本

    def is_allowed(
        self, key: str, limit: int, window_seconds: int
    ) -> tuple[bool, int]:
        client = get_redis_client()
        if client is None:
            _logger.debug("Redis 不可用,限流走 fail-open")
            return True, 0

        bucket = f"ratelimit:{key}"
        now_micro = int(time.time() * 1_000_000)
        window_micro = window_seconds * 1_000_000

        try:
            # 首次执行 SCRIPT LOAD,后续 EVALSHA 节省带宽
            if self._script_sha is None:
                self._script_sha = client.script_load(_SLIDING_WINDOW_LUA)
            t0 = time.perf_counter()
            try:
                result = client.evalsha(
                    self._script_sha, 1, bucket, now_micro, window_micro, limit
                )
            except RedisError as e:
                # NOSCRIPT — Redis 重启后脚本缓存丢了,重新 LOAD 一次
                if "NOSCRIPT" in str(e).upper():
                    self._script_sha = client.script_load(_SLIDING_WINDOW_LUA)
                    result = client.evalsha(
                        self._script_sha, 1, bucket, now_micro, window_micro, limit
                    )
                else:
                    raise
            finally:
                redis_command_latency_seconds.labels(command="EVALSHA").observe(
                    time.perf_counter() - t0
                )
        except RedisError as e:
            _logger.warning("Redis 限流操作失败,走 fail-open:%s", e)
            return True, 0

        allowed_int, retry_int = int(result[0]), int(result[1])
        return bool(allowed_int), retry_int
