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

---

## Round 2: figures, against Nature's own figure guide

Sources: Nature's figure guide, *Building and exporting figure panels*
(research-figure-guide.nature.com), and the figure requirements in the author
guidelines.

The rules that bind, quoted:

* "Figures should be laid out in a neat and space-efficient manner, minimizing
  white space and with panels in an alphabetical order wherever possible."
* "Consider the content and legibility of each panel and let that define its
  size within the figure. Some panels may require more space than others."
* Text 5 to 7 pt at final printed size, consistent across panels and across
  figures. Panel labels lowercase, 8 pt bold, not italic.
* Widths are 89 mm single column and 183 mm double; maximum height 170 mm.
* "Avoiding red/green combinations and rainbow scales helps readers with colour
  blindness to distinguish datasets."
* "Text should be used instead of decorative icons wherever possible. Icons can
  be open to interpretation and confuse the meaning of figures."
* "Keys or keylines should be used in the figure wherever possible, rather than
  having colour descriptions in the figure caption."

### Ours, checked against that

The two the author rejected fail different rules.

**Figure 13, the deblocking gate.** Panel a spends its width on a line that is
flat from 20 px onward: nine tenths of the panel carries one number. Panel b's
bars have deliberately unequal widths, because width encodes the share of
pixels, and nothing in the figure says so, which breaks the keys-in-the-figure
rule. The figure is legible and says almost nothing.

**Figure 14, the qualitative crop.** Four panels, one of which is a
near-black difference map. It is the standard "look, no difference" figure, and
its problem is not correctness but that it spends a full-width figure to show an
absence. Nature's proportionality rule says panel size should follow content;
here the most informative panel, the difference, is the least legible.

### What replaces them, and what is added

Recorded before building so the choice can be judged against the outcome:

| slot | figure | why |
|---|---|---|
| replaces 13 | the seam gate, tiled, and the correction it makes on a real frame | shows the mechanism and its effect in one place; the correction map IS the tile lattice, which is the point |
| replaces 14 | the exit map over the frame, beside what it costs | says WHERE the decoder spent, which is the paper's subject, rather than that the difference is invisible |
| new diagram | how a multiplier turns a per-tile distortion table into an exit map | the core mechanism has no diagram anywhere in the paper |
| new plot | the frontier with floor and saturation marked | the operating window is a headline claim and exists only as a table |
| new plot | concentration of regret across tiles | justifies partial signalling; currently a Gini coefficient in prose |
| new plot | the three deciders across rates, with the agreement paradox | result three of three, and it currently has no figure |

---

## Round 3: the narrative thread

Sources: Schimel, *Writing Science*, whose OCAR structure (opening, challenge,
action, resolution) is the one journal papers are built on, and the observation
that generalist journals run LDR instead, a strong lead first. Read the
introductions of ELIC (CVPR 2022) and the DCVC-UF paper alongside.

### The story this paper is telling

Stated once so the sections can be checked against it:

**Opening.** A learned decoder spends the same computation on every region of a
frame. A flat sky and a face are decoded at the same price.

**Challenge.** The obvious repair is a smaller model, and every version of that
changes the bitstream, so a file encoded yesterday cannot benefit. What is left
is not what the decoder is, but how much of it runs where.

**Action.** Cut the frame into tiles after a shared stem and let each leave the
trunk at its own depth. Three things then have to be settled: cutting is not
free, a budget only works inside a window, and something has to decide per tile.

**Resolution.** The decision is already in the file. The bits the entropy model
spent on a tile say how much computation reconstructing it needs, better than a
trained router does.

### Where the thread snaps

Checked by reading the first sentence under every heading.

**Sections 2, 3 and 5 have no opening paragraph at all.** The reader lands on
"2. Related work" and immediately on "Early exit.". Three of the paper's eight
sections begin with a subheading, so three times the thread is simply dropped
and picked up somewhere else.

**Section 4 opens on a fact, not a consequence.** "A 3x3 depthwise at feature
position (x,y) computes a weighted sum over its neighbours." The reader has just
been shown the method and is dropped into convolution arithmetic with no signal
that this section is the bill for what they were just sold.

**Two subsections open with an aside instead of the point.** 5.1 begins "One
word on the budget before the numbers" and 5.4 begins "One assumption is worth
naming before any of this". Both are worth saying and neither belongs first.
Hedging before asserting is one of the surer marks of writing that was assembled
rather than composed.

**Where it already works,** and these are the model for the rest: 3.5 opens "The
ladder and its costs are fixed by this point, and one decision is left". 5.6
opens by saying what the trained head has to beat before it is worth its cost.
5.8 opens by asking how often the map has to be recomputed, which is the
question 5.7 leaves open. Each of those hands the reader forward.

### The fix

Openings for 2, 3 and 5. Rewrite the first sentence of 4, 4.2, 5.1 and 5.4 so it
carries the thread rather than starting a new one. Move the two asides to after
the point they qualify.
