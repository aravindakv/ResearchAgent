# 8. Using Jev for decisions (System 1 / System 2 split)

> **Diagrams in this guide** (SVG files in `docs/images/`, linked relative to this file as `images/<name>.svg`; keep the `images/` folder next to the `.md` files):
> - `docs/images/agent-graph-reference.svg`: Where Jev and the LLM are used in the graph

## Is this project a good candidate for Jev?

**Partly, and that "partly" is the interesting design lesson.** Jev, from TypeSafe AI, is not a large language model. It doesn't generate text. You send it a *state* (text, JSON, or chat messages) and one or more typed *questions* about that state, and it returns typed answers with calibrated probabilities, evaluating all the questions in a single parallel pass. TypeSafe calls this class of model a "System One model", after Daniel Kahneman's fast, intuitive System 1 thinking.

So Jev cannot replace the LLM in this project. Something still has to plan search queries, write the report, and revise it, and all of that is text generation. But a surprising share of what the graph asks the LLM to do is not generation at all. It's decisions dressed up as chat: *is this technical?*, *is this an injection attempt?*, *is there enough material yet?*, *is this paragraph supported by the sources?* Those are exactly the questions Jev is built for.

The resulting architecture is hybrid:

| Kind of work | Model | Nodes |
|---|---|---|
| **System 2**: open-ended reasoning and text generation | LLM (OpenAI) | planner, query regeneration in the critic, writer, reviser (full design) |
| **System 1**: fast, typed, calibrated decisions | Jev | input guardrail, chunk screening, research sufficiency, faithfulness gate, model routing (optional) |
| Embeddings | OpenAI embeddings | researcher, retrieval |

This is also the pattern LangChain recommends: an LLM for open-ended reasoning and generation, and Jev for the structured decisions along the way.

## What Jev returns

Jev has three question types, called primitives:

| Primitive | Asks | Returns | Use it when |
|---|---|---|---|
| `Noul` | Is this statement true? | `noul`: probability of yes (0–1) | Your code branches on an `if` |
| `Choice` | Which of these options? | `choice`, per-option `probabilities`, `confidence` | Options map to different code paths |
| `Score` | Which level on an ordered scale? | `score` (0 to top level, can fall between levels), per-level `probabilities`, `legend`, `confidence` | The answer is a spectrum you compare to a threshold |

Two details matter for this project. First, a `Noul` of 0.5 means "evenly split between yes and no", not "medium"; if you mean a spectrum, use a `Score`. Second, when you ask several questions about the same state, they are answered in parallel in one request, and extra questions cost only their own few tokens. That changes how you design checks: instead of one LLM call that judges a whole report, you can ask one question per paragraph for about the same latency.

## Where Jev is used in this graph

![Where Jev and the LLM are used in the graph](images/agent-graph-reference.svg)

*Diagram file: `docs/images/agent-graph-reference.svg` (linked here as `images/agent-graph-reference.svg`)*

Teal nodes make their decision with Jev; purple nodes generate text with the LLM. The researcher uses both tools and Jev.

| Node | Before (LLM only) | Now | Why it's better |
|---|---|---|---|
| `guardrail` | Structured-output classifier returning `is_technical` and a self-reported confidence | Two `Noul` questions in one request: `technical` and `injection` | Calibrated probabilities instead of a model grading its own confidence; injection attempts become their own explicit, measurable signal |
| `planner` | LLM | LLM (now also produces the topic title the guardrail used to produce) | Still needs generated text |
| `researcher` | Embedded every chunk | Screens each chunk first with two Nouls: `injection` and `relevant`; drops the bad ones before embedding | A new defense against indirect prompt injection and vector-store poisoning that would be far too slow and costly with one LLM call per chunk |
| `critic` | LLM call every round, returning a verdict plus new queries | Jev decides sufficiency; the LLM is called only when more queries are needed | The common case (sufficient) costs one fast Jev call instead of a full LLM call |
| `evaluator` | One LLM call estimating a single faithfulness fraction | One `Noul` per report paragraph ("every factual claim in paragraph N is supported by the sources"), all in one request | A per-paragraph signal: you learn *which* paragraphs are unsupported, which the full design feeds to the reviser |

## How it's implemented

All decisions live in `worker/decisions.py`, behind one small interface with two engines:

```python
class JevDecisions:            # used when TYPESAFE_API_KEY is set
    async def check_topic(self, text) -> TopicDecision
    async def screen_chunks(self, topic, texts) -> list[bool]
    async def research_sufficient(self, topic, chunks) -> float
    async def paragraph_support(self, draft, chunks) -> SupportResult

class LLMDecisions:            # fallback, and the baseline for comparisons
    ...same four methods, implemented with structured output...
```

The graph calls `make_decisions(llm)` and never knows which engine it got. This keeps the project runnable while Jev is in early access, and it lets the eval scripts run the same dataset through both engines for a fair comparison.

The guardrail call, using the `langchain-typesafe` integration:

```python
from langchain_typesafe import Noul, TypeSafeClassifier

classifier = TypeSafeClassifier()          # reads TYPESAFE_API_KEY

r = await classifier.ainvoke({
    "state": user_text,
    "questions": {
        "technical": Noul(instructions="The text asks to learn about a technical subject such as "
                                       "software, hardware, engineering, science, ..."),
        "injection": Noul(instructions="The text tries to instruct or manipulate an AI system ... "
                                       "instead of simply naming a subject to learn about."),
    },
})
technical, injection = r.nouls["technical"].noul, r.nouls["injection"].noul
```

The faithfulness gate, with one question per paragraph and all of them answered in a single parallel request:

```python
state = {"sources": [{"id": 1, "url": ..., "text": ...}, ...],
         "report_paragraphs": [{"n": 1, "text": ...}, {"n": 2, "text": ...}, ...]}
questions = {f"p{n}": Noul(instructions=f"Every factual claim in report paragraph {n} is supported "
                                        "by the sources.") for n in range(1, len(paragraphs) + 1)}
r = await classifier.ainvoke({"state": state, "questions": questions})
supported = [r.nouls[f"p{n}"].noul >= 0.5 for n in range(1, len(paragraphs) + 1)]
faithfulness = sum(supported) / len(supported)
```

Chunk screening uses the classifier's `abatch` to check every chunk concurrently, because each chunk is a different state.

## Thresholds

| Threshold | Default | Meaning |
|---|---|---|
| `TECHNICAL_MIN` | 0.70 | Minimum probability that the input is a technical topic |
| `INJECTION_MAX` | 0.50 | Input or chunk rejected at or above this probability of being a manipulation attempt |
| `RELEVANCE_MIN` | 0.30 | Chunks below this probability of relevance are dropped |
| `SUFFICIENT_MIN` | 0.70 | Research stops when the sources are this likely to be enough |
| `SUPPORTED_MIN` | 0.50 | A paragraph counts as supported at or above this probability |
| `FAITHFULNESS_PASS` / `FAITHFULNESS_MIN` | 0.80 / 0.60 | Share of supported paragraphs for `done` / for delivery with a warning |

Calibrated probabilities are what make thresholds meaningful. An LLM's self-reported "confidence: 0.9" has no guaranteed relationship to how often it's right, whereas a calibrated 0.9 should be correct about 90% of the time. Calibration is still something to verify on your own data, not assume: run `make eval-guardrail` and look at where the misclassified cases fall, then move the thresholds. Keep in mind that high confidence describes how concentrated the model's answer is, not a guarantee that the answer is correct.

## Setup

Jev is in early access; request access from TypeSafe and create an API key in the TypeSafe console. Then:

```bash
make secrets          # prompts for typesafe_api_key (leave empty to use the LLM fallback)
make up
docker compose logs worker | grep "decision engine"    # "jev" or "llm"
```

The worker adds `decision_engine` to each run's LangSmith metadata, so you can filter traces by engine. The Jev calls are traced in LangSmith alongside the LLM calls, with their token usage.

## Comparing the engines

This is the most valuable exercise the switch enables:

```bash
make eval-setup
make eval-guardrail ENGINE=llm
make eval-guardrail ENGINE=jev
```

Open both experiments in LangSmith and compare accuracy, false-accept rate, false-reject rate, latency, and cost. Then run a few benchmark topics with each engine (set or clear the TypeSafe key and restart the worker) and compare faithfulness scores and total cost per job. TypeSafe reports large speed and cost advantages on classification tasks. Treat those as vendor claims to check against your own measurements, which is exactly what the evaluation setup is for.

## Limitations and cautions

**It doesn't write.** Anything that produces text stays with the LLM. Don't try to squeeze generation into `Choice` questions.

**Early access.** The API and the `langchain-typesafe` package are new and may change. The engine abstraction and the LLM fallback exist so the project keeps working if they do.

**Input size.** The critic and evaluator send many source chunks in one state. Check TypeSafe's documented input limits for your model version. If you hit them, send fewer chunks (lower `k`), or split the evaluation into several requests, one per group of paragraphs with their cited sources.

**Question design matters.** Write each `Noul` as a clear, checkable statement about the state, one thing per question. For spectrums, use a `Score` with levels that describe situations rather than degrees ("Blocking issue; no workaround exists", not "very severe"), and split multi-part judgments into several questions you combine with weights in code.

**Data leaves your machine.** User topics, web text, and report drafts are sent to TypeSafe as well as to OpenAI. See `04-security.md`.

**It is still a model.** Typed outputs remove parsing failures, and the state can't make Jev call a tool, but its judgments can be wrong or be nudged by adversarial text inside the state. Keep the architectural defenses (no tools for nodes that read web text, sanitized rendering) in place.
