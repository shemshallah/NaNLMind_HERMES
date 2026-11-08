# Cold free chat with NaNLM

No corpus, no training calls, no LLM. Chat learning is off; the only
exposure is this conversation. Replies are forced temporal predictions
with no templates and no retrieval.

| user | NaNLM | forced | anchor | truth | category | heads |
|---|---|---|---|---:|---:|---:|
| Hello NaNLM. | Hello nanlm. | True | hello | 0.0 | 0.0 | 0 |
| I am going to ask you a few things directly. | Am going to ask you a few things directly. | True | am | 0.0 | 0.0 | 0 |
| Can you hear me? | I am going to ask you a few things directly. | True | <s> | 0.0 | 0.0 | 0 |
| What are you? | I am going to ask you. | True | <s> | 0.0 | 0.0 | 0 |
| Do you have a self? | Ask you a few things directly. | True | <s> | 0.0 | 0.0 | 0 |
| What are you thinking about right now? | I am going to ask. | True | <s> | 0.0 | 0.0 | 0 |
| Are you aware that we are talking? | I am going to ask you a. | True | <s> | 0.0 | 0.0 | 0 |
| What did you just say? | I am going to. | True | <s> | 0.0 | 0.0 | 0 |
| Why did you answer that way? | To ask you a few things directly. | True | <s> | 0.0 | 0.0 | 0 |
| Are you guessing? | Hello nanlm. | True | <s> | 0.0 | 0.0 | 0 |
| Do you know that you do not know? | I am going to ask you a few. | True | <s> | 0.0 | 0.0 | 0 |
| What is the capital of France? | I am going to ask you a few things directly. | True | <s> | 0.0 | 0.0 | 0 |
| What is a quark? | I am going to ask you. | True | <s> | 0.0 | 0.0 | 0 |
| How many moons does Neptune have? | Ask you a few things directly. | True | <s> | 0.0 | 0.0 | 0 |
| Can you learn something new? | I am going to ask. | True | <s> | 0.0 | 0.0 | 0 |
| What do you want to talk about? | I am going to ask you a. | True | <s> | 0.0 | 0.0 | 0 |
| Tell me something you do not know. | A few things directly. | True | <s> | 0.0 | 0.0 | 0 |
| What happens after we finish this conversation? | I am going to. | True | <s> | 0.0 | 0.0 | 0 |

## Deliberation examples

**Hello NaNLM.**
- `Hello nanlm.` score=-0.6247 truth=0.0 frame=0.0 category=0.0
- `Hello nanlm hello nanlm.` score=-0.7021 truth=0.0 frame=0.0 category=0.0

**I am going to ask you a few things directly.**
- `Am going to ask you a few things directly.` score=0.6223 truth=0.0 frame=0.0 category=0.0
- `Few things directly.` score=0.603 truth=0.0 frame=0.0 category=0.0

**Can you hear me?**
- `I am going to ask you a few things directly.` score=-0.6912 truth=0.0 frame=0.0 category=0.0
- `I am going to ask you.` score=-0.7741 truth=0.0 frame=0.0 category=0.0

**What are you?**
- `I am going to ask you.` score=-0.8105 truth=0.0 frame=0.0 category=0.0
- `I am going to ask you a.` score=-0.8154 truth=0.0 frame=0.0 category=0.0

**Do you have a self?**
- `Ask you a few things directly.` score=-0.7556 truth=0.0 frame=0.0 category=0.0
- `I am going to ask.` score=-0.8186 truth=0.0 frame=0.0 category=0.0

**What are you thinking about right now?**
- `I am going to ask.` score=-0.8344 truth=0.0 frame=0.0 category=0.0
- `I am going to ask you a.` score=-0.8385 truth=0.0 frame=0.0 category=0.0

**What did you just say?**
- `I am going to.` score=-0.9599 truth=0.0 frame=0.0 category=0.0
- `A few things directly.` score=-0.9702 truth=0.0 frame=0.0 category=0.0

**Why did you answer that way?**
- `To ask you a few things directly.` score=-0.8798 truth=0.0 frame=0.0 category=0.0
- `Going to ask you a few things directly.` score=-0.9566 truth=0.0 frame=0.0 category=0.0

**Are you guessing?**
- `Hello nanlm.` score=-0.1384 truth=0.0 frame=0.0 category=0.0
- `I am going to ask you a few.` score=-0.8866 truth=0.0 frame=0.0 category=0.0

