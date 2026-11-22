"""TRACE: tolerance-typed residual programs with annealed, certified search (numeric trajectory service).

The harness depends on `traceplan.contract` and calls `TraceService.plan(request, snapshot, arm_check, ik_check)`.
"""
from .contract import (RESULTS, ArmResult, EndCandidate, Entity, SceneSnapshot, Spheres, Term, Timed, TraceRequest,
                       TraceResponse, to_jsonable)
from .service import TraceService
