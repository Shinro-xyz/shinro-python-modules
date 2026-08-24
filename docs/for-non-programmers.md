# Shinro, Explained Simply

A plain-language guide to what this software is, what it does, and why it's
built the way it is. No programming background needed.

## What this is, in one sentence

Shinro is a toolbox for **controlling moving machines** — robots, carts,
pendulums, drone-like vehicles — in a way that is modular, safe to change, and
provably correct.

## The analogy: a person driving a car

Think about what happens when you drive a car:

- **Where do you want to go?** → you pick a route (the *trajectory*).
- **Where are you right now?** → you look out the window and at the dashboard
  (the *sensors* and *estimator*).
- **How do you get there?** → you turn the wheel and press the pedals (the
  *controller*).
- **What actually moves?** → the car itself, obeying physics (the *plant*).

A self-driving car has to do all of this automatically, hundreds of times a
second. Shinro is the software skeleton that lets engineers build that loop for
many different machines. The important part: **each piece is a separate, swappable
building block.**

## The five building blocks

Everything in Shinro is one of five things:

1. **The Brain (Controller).** Decides what command to send: "push harder, we're
   falling behind," "ease off, we're overshooting." Different brain styles exist
   — some are simple and reactive, some are sophisticated and look ahead.
2. **The Body (Plant).** The machine being controlled: a wheeled base, a robotic
   arm, a cart with a pendulum on top. The body knows its current state and how
   it responds to commands.
3. **The Eyes (Estimator).** Real-world sensors are noisy and incomplete. The
   eyes fuse the noisy readings into a clean, confident picture of where the
   machine actually is.
4. **The Route Planner (Trajectory).** Turns "get from A to B" into a smooth
   path — you don't want a robot to jerk or stop abruptly.
5. **The World (Physics Engine).** An optional simulator that plays out the
   physics so you can test in software before touching real hardware.

## Why the "swappable" part matters

Imagine you bought a car and could *only* drive it to work. Then someone asks:
"can we make it take the scenic route?" or "can we put in a faster engine?"
Because the system is modular, the answers can be "yes, easily":

- want a different route planner? Swap only that piece.
- want a smarter brain? Swap only that piece.
- want to test a safer version in a simulator first? Swap the real body for the
  simulated one, keep everything else.

None of the other pieces need to be rewritten. This is the core promise of the
design: **you can change one part without breaking the others.**

## How you build a machine: recipes, not code

Building a machine with Shinro is closer to filling out a recipe card than
writing programming from scratch. Each recipe says what ingredients to use —
"use this brain, these eyes, this body, this route" — and the toolkit assembles
them for you.

Want to change the machine's behavior? Change the recipe, not the mechanics.
Want to try three different brains on the same body? Three small edits, and run
them side by side to compare. This "recipe-driven" approach is deliberate: it
keeps things predictable and lets non-programmers experiment with which
combination works best.

## The one kind of change that's scary: breaking the loop

For any of this to work, the loop has to run perfectly every tick — thousands
of times a second. A machine doesn't just need the right math once; it needs it
repeatedly and consistently, forever, without the code ever "slipping."

This is the hardest problem in practice. Humans are excellent at explaining
what they want and terrible at being perfectly consistent for 10,000 steps.

Shinro's answer is unusual and important. It **freezes the math at build time**:
once you've decided how the machine should behave, the system transcribes that
behavior into a fixed, unchanging form. It's like writing down a dance routine
once and then having a very obedient robot dance it perfectly every single time
— instead of re-improving the dancer's choreography from scratch every
performance.

## How we know it won't drift

Two layers of protection:

1. **Testing.** Before anything is trusted, it's checked against reference
   computations on hundreds of inputs, and the results must agree to within
   one billionth of one percent. The entire toolbox is put through this each
   time something changes.

2. **Proof by construction.** When the math is frozen, the frozen form is not
   re-typed from memory. It is a literal recording of the very calculations
   that were just tested. Because the recording and the original are the same,
   any error is structurally impossible — the two cannot disagree.

So the sequence is: behave correctly in the safe, flexible world (tested on
thousands of inputs) → freeze that exact behavior → deploy the frozen version
with confidence it matches.

## What it's all for, in one story

- A team wants a warehouse robot to move boxes smoothly between two points.
- They pick a body model, a route planner, eyes, and a brain from the toolbox.
- They tune it in the simulator, comparing a few brain styles.
- They freeze the best combination — the exact math — and it never changes on
  its own.
- When the hardware team needs it to behave differently, they change the recipe
  and re-freeze — quickly, safely, and in a way everyone can understand.

That's Shinro: a toolbox to make moving things behave exactly the way you want,
safely and repeatably.
