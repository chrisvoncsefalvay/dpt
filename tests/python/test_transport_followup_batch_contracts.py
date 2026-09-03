"""Invalid structural source metadata reports the public transport contract."""

from typing import Any, cast

import pytest

from dpt.transport.model import TransportError
from dpt.transport.rng import HistoryBatch


def test_malformed_source_identity_is_rejected_before_attribute_access():
    with pytest.raises(TransportError, match="source_identity"):
        HistoryBatch(17, 0, 37, source_identity=cast(Any, "fake"))
