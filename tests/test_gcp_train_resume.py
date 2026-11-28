"""The two optimizers, exploration noise and sampling state survive a learner restart."""
import numpy as np
import pytest
import torch

from lrl_train.train_gcp import Agent, arguments, read_resume, restore_learner, save


def test_full_checkpoint_next_update(tmp_path):
    args = arguments(["--preset", "all_5_families", "--device", "cpu"])
    assert (args.workers, args.bc_steps, args.dagger_steps, args.counterfactual_coef,
            args.reward_mode, args.target_kl) == (72, 73728, 110592, .25, "centered", .04)
    def learner():
        agent = Agent()
        agent.training_manifest = {}
        return agent, torch.optim.Adam(agent.parameters()), torch.optim.Adam(agent.policy.parameters())

    def step(agent, optimizer, teacher):
        for parameters, opt in [(agent.policy.parameters(), teacher), (agent.parameters(), optimizer)]:
            opt.zero_grad()
            sum(p.square().mean() for p in parameters).backward()
            opt.step()

    model, optimizer, teacher = learner()
    rng = np.random.default_rng(5)
    step(model, optimizer, teacher)
    counters = dict(steps=12, updates=2, bc_steps=25, dagger_steps=30, last_refresh=2,
                    teacher_refresh_steps=10, rollout_generation=1)
    path = tmp_path / "policy.pt"
    save(path, model, optimizer, teacher, args, rng, counters)
    state = read_resume(path, args)
    assert {k:state[k] for k in counters} == counters
    expected_np, expected_torch = rng.random(), torch.rand(3)
    step(model, optimizer, teacher)
    other, other_opt, other_teacher = learner()
    other_rng = np.random.default_rng(6)
    restore_learner(state, other, other_opt, other_teacher, other_rng)
    assert other_rng.random() == expected_np
    assert torch.equal(torch.rand(3), expected_torch)
    step(other, other_opt, other_teacher)
    assert all(torch.equal(value, other.state_dict()[key]) for key, value in model.state_dict().items())
    del state["supervised_optimizer"]
    torch.save(state, path)
    with pytest.raises(ValueError, match="full learner"):
        read_resume(path, args)
