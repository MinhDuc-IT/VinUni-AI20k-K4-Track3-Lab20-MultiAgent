# Benchmark Report

Comparison of **single-agent baseline** vs **multi-agent workflow** (Supervisor -> Researcher -> Analyst -> Writer).

## Query

> Compare single-agent and multi-agent workflows for complex research tasks

## Metrics

| Run | Latency (s) | Cost (USD) | Quality* | Citation cov. | Failure rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| single-agent | 11.43 | 0.000394 | 6.0 | 0% | 0% | tokens_in=45; tokens_out=645; sources=0; routes=n/a |
| multi-agent | 22.29 | 0.001268 | 10.0 | 100% | 0% | tokens_in=2043; tokens_out=1602; sources=5; routes=researcher,analyst,writer,done |

*Quality is a lab heuristic (length + citations + structure), not peer-review score.
Single-agent citation coverage is 0% because the baseline does not retrieve sources.

## Analysis

- Latency delta (multi - single): **+10.86s**
- Estimated cost delta: **+0.000874 USD**
- Quality heuristic delta: **+4.0**
- Citation coverage delta: **+100%**

### When multi-agent helped

- Clear role separation (search -> analysis -> writing) produced grounded citations from 5 retrieved sources.
- Shared state + route history (`researcher -> analyst -> writer -> done`) made the pipeline easier to debug.

### When single-agent was preferable

- About **2x faster** and **~3x cheaper** in estimated tokens for this query.
- Enough for a fluent overview when evidence grounding is not required.

## Failure modes observed / expected

1. **Empty / weak retrieval** - local corpus may miss niche queries; Researcher falls back to raw snippets or errors.
2. **Coordination overhead** - multi-agent makes 3 LLM calls, so latency/cost rise (seen: +10.86s).
3. **Citation drift** - Writer may cite indices not grounded in Analyst notes; Critic/heuristic coverage catches some of this.
4. **Max-iteration stop** - Supervisor ends the graph if workers fail to fill required fields before `MAX_ITERATIONS`.

**Fix approach:** tighten Researcher filtering, add Critic before done, cache search results, and keep a strong single-agent baseline for simple queries.

## Traces

- `reports/traces/single-agent_20260820T103629Z.json`
- `reports/traces/multi-agent_20260820T103629Z.json`

## Exit ticket notes

- **Use multi-agent** when the task needs retrieval + analysis + cited synthesis, or independent verification.
- **Avoid multi-agent** for short, single-hop answers where latency/cost dominate.
