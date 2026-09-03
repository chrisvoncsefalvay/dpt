"""Shared host/device identifiers for transport derivative and random domains."""

from .model import ELECTRON_REST_ENERGY_KEV as ELECTRON_REST_ENERGY_KEV
from .model import HistoryStatus

# Kernel constants and host masks derive from the same terminal status enum.
ESCAPED = int(HistoryStatus.ESCAPED)
ABSORBED = int(HistoryStatus.ABSORBED)
EVENT_BUDGET = int(HistoryStatus.EVENT_BUDGET)
CROSSING_BUDGET = int(HistoryStatus.CROSSING_BUDGET)
NUMERICAL_FAILURE = int(HistoryStatus.NUMERICAL_FAILURE)
ENERGY_SUPPORT = int(HistoryStatus.ENERGY_SUPPORT)
ANGLE_BUDGET = int(HistoryStatus.ANGLE_BUDGET)
NUMERICAL_MASK = 1 << NUMERICAL_FAILURE
ENERGY_MASK = 1 << ENERGY_SUPPORT
BUDGET_MASK = (1 << EVENT_BUDGET) | (1 << CROSSING_BUDGET) | (1 << ANGLE_BUDGET)
FAILURE_MASK = NUMERICAL_MASK | ENERGY_MASK | BUDGET_MASK
INPUT_MASK = 1

# Nonnegative values select a material-density score in the replay kernel.
SOURCE_AMPLITUDE = -1
LOG_SOURCE_AMPLITUDE = -2

# Collision events occupy the lower half; source draws and angular rejection
# reserve this bit in the event and domain words, respectively.
RANDOM_NAMESPACE_BIT = 1 << 31

# Opt-in readout terminal, outside the canonical physical status mask.
ROULETTE_TERMINATED = 8
