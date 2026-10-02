# Independent codec-bank diagram

The main paper's page-2 system figure is `independent_codec_bank.png`.
It is a conceptual AI-generated illustration, style-matched to the
existing shared early-exit figure. Six parallel lanes show alternative
complete DCVC-UF codecs with decoder depths D2/D4/D6/D8/D10/D12.
Each patch chooses one lane before encoding. The drawn modules are
schematic, not literal tensors or exact layer counts. No six-expert
router result is inferred from this image.

Generation used the existing shared-exit system figure as a style
reference. Two rejected drafts placed arrows between neighbouring codec
cards, incorrectly implying serial execution; the accepted figure has
six separate selector-to-assembly lanes.
