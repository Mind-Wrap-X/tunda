# TUNDA — Roadmap for clinicians and for people who use the companion

This document speaks to two audiences with equal care: **people seeking emotional support through voice**, and **clinicians** who may advise patients, supervise digital tools, or integrate companions into broader care.

Language is chosen to be clear, respectful, and honest about what the system can and cannot do.

---

## For people using TUNDA (patients, clients, anyone who talks with it)

### What TUNDA tries to be

- A **listener** that responds with warmth and acknowledges how you sound and what you say.
- A **companion for everyday emotional ups and downs** — not a replacement for human relationships or professional care when you need them.

### What TUNDA is not

- **Not a therapist** and not a substitute for diagnosis or treatment.
- **Not emergency services.** If you might hurt yourself or someone else, or are in immediate danger, use your **local emergency number** or go to the nearest emergency department. The app may show supportive crisis messaging; that is **not** the same as professional crisis intervention.

### How we want you to feel

- **Heard** — your words and tone matter to how the system responds.
- **Safe to stop** — you can pause or walk away anytime.
- **In control** — over time, we aim to give you clearer choices about memory (“remember this” vs “don’t store”) and tone.

### What we are improving next (for you)

- Clearer **plain-language** explanations inside the app about privacy and memory.
- **Gentler turn-taking** in voice mode so conversations feel less rushed.
- **Feedback** (“Was this helpful?”) so the companion can align better with *your* experience, not a generic label.

If something the companion says feels wrong, confusing, or upsetting, **trust your instincts** — stop using it for that moment and reach out to someone you trust or a professional.

---

## For clinicians

### Scope and positioning

TUNDA is an **affect-aware voice companion system** built on automatic speech recognition, affect *labels* from voice and/or text (path-dependent), optional retrieval-augmented memory (stronger on desktop), and large language model responses when Ollama is online — with **rule-based crisis pattern routing** and scripted grounding as first-line safeguards. Empathy is prompt- and template-guided supportive language, not validated clinical empathy.

It should be positioned to patients as:

- **Adjunctive** to care, not standalone mental health treatment.
- **Non-diagnostic**: outputs are conversational, not clinical formulations suitable for charting without human verification.

### Clinical boundaries (recommended framing with patients)

| Appropriate framing | Avoid |
|---------------------|--------|
| “A tool for reflection and emotional check-ins between sessions” | “Your therapist on your phone” |
| “May help name feelings or practice grounding language” | “Detects your diagnosis” |
| “Does not replace crisis lines or emergency care” | Implicit reliance for acute risk |

### Safety and duty of care

- Current implementation includes **keyword-pattern crisis routing** for some self-harm language. This is **not** equivalent to validated suicide-risk instruments or clinician judgment.
- **Recommend** that acute-risk patients follow your service’s existing crisis protocols; digital companions should **not** delay access to human care.
- Future roadmap items relevant to duty of care: **tiered escalation**, **region-configurable resources**, **audit-friendly logging** with appropriate **consent and retention policies**.

### Integration considerations

- **Transcripts and memories** may be sensitive PHI if deployed in healthcare contexts; deployment would require **BAA**, encryption, access controls, and jurisdiction-specific compliance — beyond the default open-source configuration.
- If you recommend TUNDA alongside therapy, consider discussing **what to share** vs **what stays private**, and **when** to use the tool vs contact you or crisis services.

### What we are improving next (for clinical credibility)

- **Evaluation harness**: repeatable dialogue scenarios and human-rated empathy metrics.
- **Stronger alignment** with motivational interviewing–style micro-skills (reflection, affirmation, summary) without claiming therapeutic modality certification.
- **Transparency**: clearer documentation of model limits, failure modes, and update cadence.

---

## Shared technical roadmap (high level)

Phases are ordered so **measurement and safety** precede **flashy features**.

1. **Trust foundation** — evaluation suite, consent-aware memory, documented crisis pathways per region.
2. **Emotional intelligence** — better fusion of voice + language + user feedback; calibration (“Was that right?”).
3. **Voice quality** — natural turn-taking, barge-in, prosody-aware TTS where feasible.
4. **Clinical-grade deployment (optional fork)** — hosting, PHI handling, and institutional review where required.

---

## Closing note

For **patients**: your wellbeing comes first. Tools like this should **support** you, never isolate you from real help.

For **clinicians**: we welcome feedback that keeps the product **honest, bounded, and useful** without overstating capability.

If you have suggestions for this document or for safer defaults, please open an issue or contribute through your usual project channels.
