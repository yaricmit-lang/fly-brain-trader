# FlyBrain Trader: a plain-language overview

Scientists have scanned the brain of a real fruit fly, mapping essentially
every cell and every connection between them. We took that map and brought it
to life on a computer, then taught it to play the cryptocurrency market. Not
for real (the money is simulated; it is a training ground), but as an
experiment: can a fly's brain find patterns in prices if it is trained the
right way?

## How it works, in simple terms

**A real fly has eyes and a nose; our fly gets numbers instead.** Every hour
it "sniffs" the prices of five assets: Bitcoin, Ether, Solana, gold, and ZEC.
For each one it looks not only at the price itself but also at a variety of
cues that traders commonly use. For Ether, for example, it looks at classic
"waves" in the chart; for Bitcoin, at the overall direction of movement (the
trend); for gold, at small back-and-forth fluctuations.

**The brain is real, but only a small piece of it.** Not the whole fly brain,
only the part that, in real insects, handles memory and learning through
reward: "if it went well, remember what led to it". It has about 6,000 cells
and 100,000 connections between them, and every connection is one that
scientists found in a real fly, not an invented one.

**The fly does not decide whether to buy or sell.** This is deliberate: we
tested it, and it turned out that this data simply does not allow reliably
guessing whether a price will rise or fall, which is like flipping a coin. It
is, however, possible to estimate fairly well how "nervous" (volatile) the
market is right now. So a simple formula decides the direction and the default
amount to invest, and the fly's brain learns just one thing: how to be a bit
bolder or a bit more cautious at this moment, depending on the circumstances.

**It learns from its own successes and mistakes.** If a decision turns out to
be profitable some time later, the connections that led to it are strengthened
slightly, as if the brain had received a dose of a "pleasure hormone"
(dopamine, as in real animals). If it lost money, they are weakened instead.
The reward does not arrive immediately but after 12 hours, which is just long
enough to tell what that particular decision led to, rather than the next one.

**There are safeguards** so the brain cannot accidentally lose the whole
capital: a limit on how much can go into one asset, a stop for the day if the
loss gets too large, and a cap on the number of trades per hour.

**You can watch it all in real time.** There is a browser page: on one side, a
3D model of the brain spins, with points flashing green when the brain
receives "praise" and red when it receives "punishment". On the other side are
the price charts and what the fly is currently thinking and doing.

## So does it make money?

In short: no, not consistently. We checked this honestly over several years of
history. The fly trades about as well as a simple formula with no "brain" at
all: sometimes slightly better, sometimes slightly worse, but without a clear
and reliable edge. The wiring of the real fly brain also gave no noticeable
gain compared with randomly shuffled connections of the same size. It seems
that a brain tuned by nature for smells need not be well suited to market
charts. This is a normal, honest scientific result: we tested it and did not
hide the inconvenient numbers (they are all recorded in METHODOLOGY.md). The
value of the project today is a working, vivid experiment, not a way to make
money.

## What is in the project (for anyone who opens the folder)

- **config/**: all settings in one place: which coins, which risk rules, and so on
- **src/flybrain/**: all the code for the brain, training, and trading
- **server/** and **web/**: the dashboard page you open in the browser
- **data.nosync/** and **runs.nosync/**: downloaded data and training results (not kept in git or cloud sync; too heavy)
- **tests/**: automated checks that everything is computed correctly

## How to run it

```bash
python -m flybrain.serve --adapter dry_run --online-learning --auto-approve
```

Then open `http://127.0.0.1:8000` in a browser to see the live brain. The
different training variants are launched with buttons on that same page.

## For more detail

- [README.md](../README.md): a short description for anyone who just stopped by the repository
- [DESIGN_LOG.md](DESIGN_LOG.md): the engineering decisions and the pitfalls hit along the way
- [METHODOLOGY.md](METHODOLOGY.md): the full methodology and numbers, for those who want to verify the conclusions
