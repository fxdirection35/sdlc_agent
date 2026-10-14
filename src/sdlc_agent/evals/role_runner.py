"""Bridge between golden tasks and role implementations.

The runner in :mod:`sdlc_agent.evals.runner` scores a finished
:class:`AgentRun`; this module produces those runs by driving a
:class:`~sdlc_agent.pipeline.roles.Role` implementation with a golden task.
That separation is what lets the same eval set score the scripted roles of P1,
the litellm roles of P2 and whatever comes later, unchanged.

Not every golden task is role-addressable. A role produces exactly one
artifact kind; a task whose type checks name a different kind (QA's
"report an honest not-run" task expects a ``TestRun``, which the executor
produces, not the QA role) targets a different stage of the pipeline. Such
tasks are excluded *with a reason string* rather than silently dropped - an
eval set that quietly runs fewer tasks than it appears to is the failure mode
``load_tasks`` refuses elsewhere in this package.
"""

from __future__ import annotations

from collections.abc import Sequence

from sdlc_agent.domain.base import ArtifactKind, PipelineRole
from sdlc_agent.domain.product import Requirement
from sdlc_agent.evals.runner import AgentRun, EvalReport, run_eval_set
from sdlc_agent.evals.tasks import GoldenTask, KindCorrectCheck, SchemaValidCheck
from sdlc_agent.pipeline.roles import (
    ROLE_OUTPUTS,
    Role,
    RoleContext,
    RoleError,
)

#: Stamped on the synthetic requirement so a run is traceable to its task.
#: Not a real remote: no repository is cloned in an eval, and pretending one
#: was would be a small lie in exactly the artifact whose job is fidelity.
_EVAL_SOURCE = "eval://golden/{task_id}"

#: The requirement title is capped at 200 characters by the model; the first
#: line of the prompt is the natural title and never exceeds that in practice,
#: but the cap is enforced here so a long-winded prompt cannot fail validation.
_TITLE_CAP = 200


def expected_kinds(task: GoldenTask) -> frozenset[ArtifactKind]:
    """The artifact kinds a task's type checks demand."""
    return frozenset(
        check.expected_kind
        for check in task.checks
        if isinstance(check, SchemaValidCheck | KindCorrectCheck)
    )


def exclusion_reason(task: GoldenTask) -> str:
    """Why no role implementation can serve this task; ``''`` when one can.

    The reason exists so reports and tests can name what was not run. A task
    with no type checks constrains nothing about kind and is always runnable -
    the checks may still fail, which is the eval doing its job.
    """
    output_kind = ROLE_OUTPUTS[task.role]
    others = sorted(kind for kind in expected_kinds(task) if kind is not output_kind)
    if not others:
        return ""
    wanted = " or ".join(kind.value for kind in others)
    return (
        f"expects {wanted} but the {task.role.value} role produces "
        f"{output_kind.value}; this task targets another pipeline stage"
    )


def role_addressable_tasks(
    role: PipelineRole, tasks: Sequence[GoldenTask]
) -> tuple[GoldenTask, ...]:
    """The tasks an implementation of ``role`` can run, in input order."""
    return tuple(task for task in tasks if task.role is role and not exclusion_reason(task))


def context_for_task(task: GoldenTask) -> RoleContext:
    """Build the role context a golden task runs under.

    The prompt becomes the requirement description verbatim - the eval measures
    the role against the requester's words, so nothing here may tidy the prompt
    up. The context is otherwise pristine: first attempt, no feedback, no
    upstream artifacts. That determinism is what lets a recorded response be
    replayed: the same task always produces the same request shape.
    """
    return RoleContext(
        requirement=Requirement.model_validate(
            {
                "title": _title_from(task),
                "description": task.prompt,
                "requester": "eval-harness",
                "source_repo": _EVAL_SOURCE.format(task_id=task.task_id),
            }
        )
    )


def _title_from(task: GoldenTask) -> str:
    """First non-blank line of the prompt, capped to the model's limit."""
    for line in task.prompt.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:_TITLE_CAP]
    # prompt is reject_blank-validated, so this is unreachable; keeping it
    # total means _title_from cannot raise its way out of a build_context call.
    return task.task_id[:_TITLE_CAP]


def run_golden_task(role_impl: Role, task: GoldenTask) -> AgentRun:
    """Run one task through one role implementation.

    A :class:`RoleError` becomes an artifact-less run rather than an exception:
    a role that cannot answer is an evaluable outcome - every artifact check
    fails with "agent produced no artifact" - while re-raising would turn a weak
    agent into a broken eval run and send whoever investigates to the wrong
    layer.
    """
    try:
        result = role_impl.run(context_for_task(task))
    except RoleError:
        return AgentRun(artifact=None)
    # Constructed here rather than via a method on RoleResult so the pipeline
    # layer never imports the eval layer: evals adapts to pipeline, not the
    # reverse (importing pipeline.roles as the entry module used to crash on
    # the cycle role_runner -> pipeline.roles -> evals.runner -> role_runner).
    return AgentRun(
        artifact=result.artifact,
        raw_payload=result.raw_payload,
        output_tokens=result.output_tokens,
        tools_called=result.tools_called,
    )


def run_role_eval_set(role_impl: Role, tasks: Sequence[GoldenTask]) -> EvalReport:
    """Score one role implementation against its addressable golden tasks.

    Tasks for other roles are not this call's business; excluded same-role
    tasks are already filtered out by :func:`role_addressable_tasks`, and the
    empty report that zero addressable tasks would produce is deliberately not
    green (see :attr:`EvalReport.is_green`).
    """
    role = role_impl.role
    runs = {
        task.task_id: (task, run_golden_task(role_impl, task))
        for task in role_addressable_tasks(role, tasks)
    }
    return run_eval_set(runs)
