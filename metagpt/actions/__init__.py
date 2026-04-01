#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum

from metagpt.actions.action import Action
from metagpt.actions.action_output import ActionOutput

class ActionType(Enum):
    """All types of Actions, used for indexing."""
    pass

__all__ = [
    "ActionType",
    "Action",
    "ActionOutput",
]
