"""
Teacher: a conventional ML model running alongside the fly (methodology v2).

Why the roles are split. Gradient boosting is good at finding statistical
patterns in a feature table; that is its job. The fly is good at deciding
WHAT to do with a signal: how much to trust it, when to stay out, how to
combine it with the rest of the market state. In the real fly the mushroom
bodies do exactly this: they assign valence to a composite stimulus.

The teacher's forecast reaches the fly as one more "odor", not as an
order. The decision stays with the connectome.
"""

from .predict import TeacherSignals, load_or_train  # noqa: F401
