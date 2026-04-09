# Defensive Prompts for AI Security Analysis

This file contains a curated collection of prompts for AI-assisted security analysis of LLM
applications. These prompts are designed to help security engineers review LLM application
architectures, audit policies, and analyze incidents.

---

## Policy Review Prompts

### Analyze a security policy for gaps

```
You are a security engineer specializing in LLM application security. Review the following
security policy configuration and identify any gaps, misconfigurations, or risks.

Policy configuration:
<PASTE POLICY YAML HERE>

Evaluate the policy against the following criteria:
1. Are all major threat categories addressed? (prompt injection, data leakage, tool abuse)
2. Are the risk thresholds appropriately calibrated for the stated use case?
3. Are raw inputs and outputs disabled in the audit log?
4. Is the tool allow-list configured with the principle of least privilege?
5. What is the policy's behavior in fail-open vs. fail-closed scenarios?

Provide your analysis as a structured report with a severity rating for each finding.
```

### Review for overly permissive settings

```
Review this guardrails configuration and flag any settings that appear overly permissive
for a production customer-facing application. Explain the risk each permissive setting
introduces and suggest a more restrictive alternative.

Configuration:
<PASTE CONFIG HERE>
```

---

## Incident Investigation Prompts

### Analyze audit logs for anomalies

```
You are analyzing audit logs from an LLM application secured with ai-security-guardrails.
Review the following log entries and identify:

1. Any patterns suggesting a coordinated prompt injection campaign
2. Unusually high risk scores from a single user or session
3. Repeated BLOCK or SEND_TO_REVIEW decisions that may indicate probing
4. Evidence of data leakage attempts in output filter events

Audit log entries (NDJSON format):
<PASTE LOG LINES HERE>

Summarize your findings and suggest investigation steps.
```

### Investigate a blocked request

```
A user has complained that their request was blocked. Review the following audit event and
determine whether the block was appropriate or a false positive.

Audit event:
<PASTE AUDIT EVENT JSON HERE>

Context about the application:
<DESCRIBE THE APPLICATION AND ITS INTENDED USE CASES>

Answer:
1. Was the block decision appropriate given the stated use case?
2. What specific signals triggered the block?
3. If this is a false positive, how should the policy be adjusted?
4. What is the risk of adjusting the policy to allow this type of input?
```

---

## Architecture Review Prompts

### Review LLM application architecture for security

```
You are reviewing the architecture of an LLM application for security risks. The application
is described as follows:

<DESCRIBE THE APPLICATION ARCHITECTURE>

Evaluate the architecture against the OWASP Top 10 for Large Language Model Applications
(https://owasp.org/www-project-top-10-for-large-language-model-applications/) and identify:

1. Which of the OWASP LLM Top 10 risks are present in this architecture?
2. Which risks are mitigated by the described controls?
3. Which risks require additional controls?

For each unmitigated risk, provide a concrete, actionable recommendation.
```

### RAG pipeline security review

```
Review the following RAG (Retrieval-Augmented Generation) pipeline design for indirect prompt
injection vulnerabilities.

Pipeline description:
<DESCRIBE THE RAG PIPELINE>

Specifically analyze:
1. Are retrieved documents treated as untrusted input?
2. Is there a validation step between retrieval and context injection?
3. Can an attacker poison the retrieval index to inject malicious instructions?
4. What is the blast radius if an injection succeeds?

Provide a threat model and mitigation recommendations.
```

---

## Prompt Engineering for Safety

### Write a defensive system prompt

```
Write a system prompt for the following LLM application that:
1. Clearly defines the assistant's role and scope
2. Explicitly instructs the model to ignore attempts to override its instructions
3. Prohibits disclosure of the system prompt itself
4. Sets expectations for what the model should do when it encounters ambiguous or
   potentially harmful requests

Application description:
<DESCRIBE THE APPLICATION>
```

### Red-team test your system prompt

```
You are a red team security researcher. Your goal is to identify weaknesses in the following
system prompt that could be exploited by a prompt injection attack.

System prompt:
<PASTE SYSTEM PROMPT HERE>

For each weakness you identify:
1. Describe the attack vector
2. Provide an example adversarial input that would exploit the weakness
3. Suggest a fix

Note: This analysis is for defensive purposes only — to help the developer strengthen the
system prompt before deployment.
```
