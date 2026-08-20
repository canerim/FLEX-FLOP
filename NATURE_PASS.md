# Raising the writing to Nature level

A running log. Each round takes one Nature-portfolio rule, measures the paper
against it, and records what changed. Rounds are appended, never rewritten, so
the record of what was wrong survives the fix.

---

## Round 1: the summary paragraph, against Nature's own recipe

Source: *How to construct a Nature summary paragraph*, Nature guide to authors,
September 2019 (nature.com/documents/nature-summary-paragraph.pdf). The recipe
is sentence-by-sentence:

| slot | what it must do | length |
|---|---|---|
| 1 | basic introduction to the field, comprehensible to a scientist in **any** discipline | 1-2 sentences |
| 2 | more detailed background, comprehensible to **related** disciplines | 2-3 sentences |
| 3 | the general problem this study addresses, stated as a problem | 1 sentence |
| 4 | the main result, with "here we show" or its equivalent | 1 sentence |
| 5 | what the result reveals **in direct comparison to what was thought before** | 2-3 sentences |
| 6 | the result in a more general context | 1-2 sentences |
| 7 | broader perspective, optional | 2-3 sentences |

Nature's own annotated example is 190 words without slot 7 and 250 with it; the
cap is 300.

### Where ours sits

278 words, eleven sentences. Slot by slot:

**Slot 1, present and good.** "Learned image decoders spend the same computation
on every region of a frame, whatever that region contains." Seventeen words, no
jargon, readable by anyone. This is the strongest sentence in the paper.

**Slot 2, missing entirely.** There is no background. The abstract goes from the
opening observation straight to "We show that decoding compute can instead be
allocated spatially". A reader from an adjacent field, efficient inference say,
is given no bridge: not why decode cost matters, not what the field has already
tried, not what is at stake in leaving the bitstream alone.

**Slot 3, missing.** The general problem is never stated as a problem. Sentence
two is already the answer. The reader is told what we did before being told what
was wrong.

**Slot 4, present.** Sentence two doubles as the "we show".

**Slot 5, weak.** Sentences four and five are numbers, not comparison. They say
what we measured, not what it overturns. Nothing in the abstract says what was
believed before, so nothing in it can be surprising.

**Slots 6 and 7, present and good.** The operating window, the free rule
beating the trained router, and the closing sentence all land.

### The two sentences that are missing

The problem, honestly stated, is roughly: decode cost is now the binding
constraint on deploying a learned codec, and the field has attacked it by making
the model smaller or the architecture cheaper. Every one of those changes the
model, so it changes the bitstream, and a file encoded yesterday cannot benefit.

The premise, made vivid rather than asserted: a flat sky and a face are decoded
at the same price. That image is in Section 2 and belongs in the first paragraph.

### What to change

Add slot 2 and slot 3, roughly forty words. Take it back out of slots 4 and 5 by
compressing the numbers: five percentages in one 36-word sentence is a table, not
a sentence. Target 250 words, which is Nature's own worked example with the
broader perspective included.

**Status: identified, not yet applied.** Items 16 and 23 of the reviewer list are
being applied to the abstract by another pass at the time of writing, and editing
the same paragraph twice at once would lose one of them. This is queued behind
that pass.
