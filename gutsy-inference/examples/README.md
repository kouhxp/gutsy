# Examples

Six small browser demos. The data in them (the ticket, feedback rows, the source document,
the call transcript) is made up; every probability on screen comes from your local gutsy server.

## Run

Start the server with CORS on, so a page opened from disk may call it:

    cd gutsy-inference
    gutsy-inference serve --cors

Then open any of the HTML files in a browser. Each file is self-contained; nothing else to install.
If the server runs elsewhere, add `?server=http://host:port` to the page URL. If the browser asks
whether the page may access devices on your local network, allow it.

`--cors` lets any web page you visit query the server while it runs, so leave it off when you're
not using the examples.

<!--
Side-by-side preview. Record a short GIF of each demo into examples/gifs/ and uncomment.

| Support triage | Feedback sheet | Model router | Fact check | Call radar | Maze autoplay |
|---|---|---|---|---|---|
| ![](gifs/support-triage.gif) | ![](gifs/feedback-sheet.gif) | ![](gifs/model-router.gif) | ![](gifs/fact-check.gif) | ![](gifs/call-radar.gif) | ![](gifs/maze-autoplay.gif) |
-->

## The demos

**[support-triage.html](support-triage.html)**: one support ticket and a "Triage ticket" button.
gutsy answers three questions about it in a single request: which team should handle it (a
`choice` over billing, technical, account and sales, with confidence and margin), whether the
customer asks for money back (a `noul`; the probability itself is the confidence), and how upset
the customer is (a 4-level `score` from calm to furious).
`state` is the ticket as text; the three questions share it, so it is encoded once.

**[feedback-sheet.html](feedback-sheet.html)**: 30 rows of customer feedback for an invoicing
app. Type any yes/no question (or click a suggestion such as "Is this a churn risk?" or "Mentions a
competitor?") and gutsy answers it for every row; each question becomes a column of P(yes). Click
a column header to sort by it.
One request per row: `state` is the row as JSON, one `noul` question with your text.

**[model-router.html](model-router.html)**: a chat box that shows which model a message would be
sent to: small model, large model with reasoning off, or large model with a high reasoning
budget. No reply is generated; the routing decision is the demo.
One request per message: `state` is `{"user_message": ...}`, one `choice` question over the three
routes, with a description of each.

**[fact-check.html](fact-check.html)**: a 1,500-word report and an LLM's answer about it, with
some claims wrong on purpose. Press "Fact check claims" and each answer sentence gets P(supported),
shaded from red to green, plus a link to the passage gutsy picked as most relevant. Click a
sentence to jump to its passage.
`state` is the report with numbered passages (sent once and cached by the server); per sentence,
one `noul` ("is this claim supported?") and one `choice` over the 15 passages.

**[call-radar.html](call-radar.html)**: plays a sales call transcript as if live. After each
turn, gutsy checks it for an objection (and its type), a competitor (and which one), a
commitment, and the customer's confusion level. Triggers drop a marker on the timeline and push a
suggested card to the rep. Playback waits for gutsy after each turn, so the radar never falls
behind the call.
`state` is the call so far plus the latest turn. Customer turns ask `noul` + `choice` for
objections and competitors, a `noul` for commitments and a 4-level `score` for confusion; rep turns
ask only about commitments. A trigger fires at P(yes) >= 0.6; "confusion rising" fires when the
expected confusion level climbs by 0.3 or more and is at least 1.

**[maze-autoplay.html](maze-autoplay.html)**: a small maze chase (original characters) that gutsy
plays by itself. At each junction the page describes every way out in words (how far the nearest
enemy would be, how many crumbs lie that way, whether it eats one now, whether it keeps going
straight) and asks a `choice` question; gutsy's pick is the move. Two guardrails sit around the
model, as you would put around any automated decision: moves that step next to an enemy are not
offered when a safe one exists, and turning back is only offered when an enemy blocks the way, which
also stops back-and-forth loops. In a corridor that leaves one way forward, so the page keeps moving
without asking ("forced" in the panel). Enemies move only after gutsy answers, so CPU speed doesn't
matter. Pause autoplay to play with the arrow keys.

The suggestion cards in the call radar and the competitor names are canned text for the demo;
gutsy decides which card to show.
