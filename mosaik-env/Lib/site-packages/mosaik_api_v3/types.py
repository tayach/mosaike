from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple
from typing_extensions import Literal, TypeAlias, TypedDict


Time: TypeAlias = int
"""Time is represented as the number of simulation steps since the
simulation started. One step respresents `time_resolution` seconds."""
Attr: TypeAlias = str
"""An attribute name"""
ModelName: TypeAlias = str
"""The name of a model."""

ModelDescriptionOptionals = TypedDict(
    "ModelDescriptionOptionals",
    {
        "any_inputs": bool,
        "trigger": Iterable[Attr],
        "non-trigger": Iterable[Attr],
        "persistent": Iterable[Attr],   
        "non-persistent": Iterable[Attr],
    },
    total=False,
)

class ModelDescription(ModelDescriptionOptionals):
    """Description of a single model in `Meta`"""
    public: bool
    """Whether the model can be created directly."""
    params: List[str]
    """The parameters given during creating of this model."""
    attrs: List[Attr]
    """The input and output attributes of this model."""

class MetaOptionals(TypedDict, total=False):
    extra_methods: List[str]
    """The names of the extra methods this simulator supports."""

class Meta(MetaOptionals):
    """The meta-data for a simulator."""
    api_version: Literal["3.0"]
    """The API version that this simulator supports in the format "major.minor"."""
    type: Literal['time-based', 'event-based', 'hybrid']
    """The simulator's stepping type."""
    models: Dict[ModelName, ModelDescription]
    """The descriptions of this simulator's models."""

SimId: TypeAlias = str
"""A simulator ID"""
EntityId: TypeAlias = str
"""An entity ID"""
FullId: TypeAlias = str
"""A full ID of the form "sim_id.entity_id" """
InputData: TypeAlias = Dict[EntityId, Dict[Attr, Dict[FullId, Any]]]
"""The format of input data for simulator's step methods."""
OutputRequest: TypeAlias = Dict[EntityId, List[Attr]]
"""The requested outputs for get_data. For each entity where data is
needed, the required attributes are listed."""
OutputData: TypeAlias = Dict[EntityId, Dict[Attr, Any]]
"""The format of output data as return by ``get_data``"""

class CreateResultOptionals(TypedDict, total=False):
    rel: List[EntityId]
    """The entity IDs of the entities of this simulator that are
    related to this entity."""
    children: List[CreateResult]
    """The child entities of this entity."""
    extra_info: Any
    """Any additional information about the entity that the simulator
    wants to pass back to the scenario.
    """

class CreateResult(CreateResultOptionals):
    """The type for elements of the list returned by `create` calls in
    the mosaik API."""
    eid: EntityId
    """The entity ID of this entity."""
    type: ModelName
    """The model name (as given in the simulator's meta) of this entity.
    """

CreateResultChild: TypeAlias = CreateResult

class EntitySpec(TypedDict):
    type: ModelName

class EntityGraph(TypedDict):
    nodes: Dict[FullId, EntitySpec]
    edges: List[Tuple[FullId, FullId, Dict]]
