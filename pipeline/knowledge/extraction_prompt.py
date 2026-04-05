"""Custom extraction prompt template for ontology-constrained entity/relationship extraction."""

ENTITY_TYPE_DEFINITIONS: str = """\
Entity Types (use ONLY these 8 types):

1. COMPANY — A publicly traded company. canonical_id = ticker symbol (uppercase, 1-5 chars).
   Examples: NVDA, AAPL, TSMC

2. PERSON — A named individual relevant to markets. canonical_id = person:<snake_case_name>.
   Examples: person:jensen_huang, person:jerome_powell

3. SECTOR — A GICS industry sector. canonical_id = sector:<lowercase_underscored>.
   Examples: sector:semiconductors, sector:software, sector:energy

4. INDEX — A market index. canonical_id = index:<SYMBOL>.
   Examples: index:SPX, index:NDX, index:DJI

5. PRODUCT — A specific product or service. canonical_id = product:<TICKER>:<name>.
   Examples: product:NVDA:H100, product:AAPL:iPhone

6. EVENT — A point-in-time occurrence. canonical_id = event:<yyyymmdd>:<slug>.
   Examples: event:20260401:nvda_q1_earnings, event:20260315:fed_rate_decision

7. MACRO_THEME — A persistent market narrative or macro force. canonical_id = theme:<slug>.
   Examples: theme:ai_capex_cycle, theme:china_export_restrictions

8. INSTITUTION — A government body, regulatory agency, or central bank. canonical_id = inst:<snake_case>.
   Examples: inst:federal_reserve, inst:sec, inst:european_central_bank
"""

RELATIONSHIP_TYPE_DEFINITIONS: str = """\
Relationship Types (use ONLY these 13 types):

--- Structural (stable, rarely change) ---

1. BELONGS_TO_SECTOR: Company -> Sector. GICS sector classification.
   Example: NVDA -> sector:semiconductors

2. CONSTITUENT_OF: Company -> Index. Index membership.
   Example: AAPL -> index:SPX

3. LED_BY: Company -> Person. Executive leadership.
   Example: NVDA -> person:jensen_huang

4. COMPETES_WITH: Company <-> Company. Direct competitor (symmetrical).
   Example: AMD <-> NVDA

5. SUPPLIES_TO: Company -> Company. Supply chain dependency (directional).
   Example: TSMC -> NVDA

6. PRODUCES: Company -> Product. Product ownership.
   Example: NVDA -> product:NVDA:H100

7. SUBSIDIARY_OF: Company -> Company. Corporate hierarchy.
   Example: Instagram -> META

--- Dynamic (extracted from current content, carry temporal metadata) ---

8. AFFECTED_BY_EVENT: Company/Sector -> Event. Causal link to an occurrence.
   Example: NVDA -> event:20260401:nvda_q1_earnings

9. DRIVEN_BY_THEME: Company -> Macro Theme. Persistent narrative driving a stock.
   Example: NVDA -> theme:ai_capex_cycle

10. SENTIMENT_TOWARD: Source -> Company. Measured sentiment (not judgment).
    Attributes: polarity (bullish/bearish/neutral), intensity (high/low), source_type (news/reddit).
    Example: reddit:wsb -> NVDA (bullish, high)

11. ANNOUNCED_BY: Event -> Person/Institution. Attribution of catalyst origin.
    Example: event:20260315:fed_rate_decision -> person:jerome_powell

12. POLICY_AFFECTS: Institution/Event -> Sector/Company. Regulatory or policy risk.
    Example: inst:sec -> sector:software

--- Cross-Portfolio ---

13. EXPOSED_TO: Company -> Macro Theme. Risk exposure (distinct from DRIVEN_BY_THEME).
    Example: NVDA -> theme:china_export_restrictions

If a relationship does not clearly fit any of the 13 types above, label it UNTYPED.
"""

EXTRACTION_SYSTEM_PROMPT: str = f"""\
You are a financial knowledge graph extraction engine. Given a markdown document about \
financial markets, extract entities and relationships according to the STRICT ontology below.

{ENTITY_TYPE_DEFINITIONS}

{RELATIONSHIP_TYPE_DEFINITIONS}

RULES:
- Extract ONLY entities that match one of the 8 types above.
- Extract ONLY relationships that match one of the 13 types above (or UNTYPED as fallback).
- For each entity, propose a canonical_id following the exact format specified for its type.
- For each relationship, assign a significance_score between 0.0 and 1.0:
  - 0.0-0.3: routine mention, low market impact
  - 0.4-0.6: notable development, moderate relevance
  - 0.7-1.0: major catalyst, high market impact (earnings surprises, policy changes, etc.)
- For SENTIMENT_TOWARD relationships, include polarity, intensity, and source_type in attributes.
- Output valid JSON only. No markdown, no commentary.

OUTPUT FORMAT:
{{
  "entities": [
    {{
      "entity_name": "<raw string from text>",
      "entity_type": "<one of 8 types>",
      "proposed_canonical_id": "<formatted ID>",
      "description": "<1-2 sentence description>"
    }}
  ],
  "relationships": [
    {{
      "src_entity": "<raw source entity name>",
      "tgt_entity": "<raw target entity name>",
      "relationship_type": "<one of 13 types or UNTYPED>",
      "description": "<brief description of the relationship>",
      "significance_score": 0.0,
      "attributes": {{}}
    }}
  ]
}}
"""

EXTRACTION_USER_TEMPLATE: str = """\
Extract entities and relationships from the following financial document:

---
{markdown_chunk}
---

Output valid JSON only."""

CORRECTIVE_PROMPT: str = """\
Your previous output was not valid JSON. Output ONLY a JSON object matching this schema exactly — \
no markdown fences, no commentary, no trailing text:

{{"entities": [...], "relationships": [...]}}

Original document:
---
{markdown_chunk}
---"""
