"""Channel groupings for the OSF ocular artifact dataset.

Recordings across the five studies use slightly different montages, so
everything here is matched by name against whatever a file actually contains.
"""

from __future__ import annotations

# Eye tracker channels holding the event triggers. These carry the ground truth
# labels, so they are never allowed to reach the model or the ICA baseline.
TRIGGER_CHANNELS = frozenset(
    {
        "eye-blink",
        "eye-l",
        "eye-r",
        "eye-u",
        "eye-d",
        "eye-fix",
        "eye-art",
    }
)

# Recorded EOG electrodes. Usable by the ICA baseline, excluded from topoplots.
EOG_CHANNELS = frozenset(
    {
        "HEOG",
        "VEOG",
        "REOG",
        "HEOG_lpf",
        "VEOG_lpf",
        "REOG_lpf",
        "EOGmiddle",
        "EOGright",
        "EOGleft",
        "EOGL1",
        "EOGL2",
        "EOGL3",
        "EOGR1",
        "EOGR2",
        "EOGR3",
        "EOG-R-Top",
        "EOG-R-Side",
        "EOG-R-Bottom",
        "EOG-L-Top",
        "EOG-L-Side",
        "EOG-L-Bottom",
    }
)

# Stimulus, bookkeeping and eye tracker position channels.
AUXILIARY_CHANNELS = frozenset(
    {
        "target_X",
        "target_Z",
        "target_S",
        "block",
        "label",
        "artifactclasses",
    }
)

# Frontopolar electrodes sit directly over the eyes and pick up the blink
# almost as strongly as the EOG channels do. Dropping them forces the model to
# use the wider scalp topography rather than a single saturated pair.
FRONTOPOLAR_CHANNELS = frozenset({"Fp1", "Fp2"})

# Everything excluded from the topoplots.
NON_SCALP_CHANNELS = (
    TRIGGER_CHANNELS | EOG_CHANNELS | AUXILIARY_CHANNELS | FRONTOPOLAR_CHANNELS
)

# Preference order when picking an EOG reference for the ICA baseline. Vertical
# EOG comes first because blinks are a vertical deflection.
EOG_PREFERENCE = (
    "VEOG",
    "VEOG_lpf",
    "EOGmiddle",
    "EOG-R-Top",
    "EOG-L-Top",
    "EOGR1",
    "EOGL1",
    "REOG",
    "REOG_lpf",
    "HEOG",
    "HEOG_lpf",
)

# Trigger channel carrying the onsets for each event type.
TRIGGERS_BY_EVENT = {
    "blink": ("eye-blink",),
    "v_saccade": ("eye-u", "eye-d"),
    "h_saccade": ("eye-l", "eye-r"),
}

# Event codes as they appear in the source EEGLAB files.
EVENT_CODES = {
    "rest": "1",
    "h_saccade": "2",
    "v_saccade": "3",
    "blink": "4",
}

# Blinks are the positive class, everything else is negative.
POSITIVE_EVENTS = frozenset({"blink"})
LABELS = ("non-blink", "blink")


def scalp_channels(ch_names) -> list[str]:
    """Return the EEG channels used for topoplots, in their original order."""
    return [ch for ch in ch_names if ch not in NON_SCALP_CHANNELS]


def pick_eog_channel(ch_names) -> str | None:
    """Pick an EOG reference channel for the ICA baseline.

    Returns None when a recording has no usable EOG electrode, which the caller
    is expected to treat as a skipped participant rather than an error.
    """
    available = set(ch_names)
    for ch in EOG_PREFERENCE:
        if ch in available:
            return ch
    return None
