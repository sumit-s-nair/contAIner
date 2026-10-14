import math
import torch
import pytest

from scripts.train_system2_grpo import compute_per_token_losses


def test_pg_gradient_non_zero_when_diverged_from_ref():
    """
    (a) With logp - ref_logp = log(2) (pi/ref = 2) and advantage +1,
    the PG gradient w.r.t. logp is non-zero (-1 per token).
    """
    ref_logp = torch.tensor([[0.5, -1.2, 0.0]])
    logp = (ref_logp + math.log(2.0)).clone().detach().requires_grad_(True)
    advantages = torch.tensor([1.0])

    pg_loss, _ = compute_per_token_losses(logp, ref_logp, advantages, clip_eps=0.2)
    pg_loss.sum().backward()

    assert logp.grad is not None
    expected_grad = torch.full_like(logp, -1.0)
    assert torch.allclose(logp.grad, expected_grad)


def test_kl_values_match_k3_formula():
    """
    (b) The KL value equals exp(-x) + x - 1 at a few hand-picked x values,
    and is 0 at x = 0.
    """
    # At x = 0, KL must be 0
    ref_zero = torch.tensor([[0.0]])
    logp_zero = torch.tensor([[0.0]])
    _, kl_zero = compute_per_token_losses(logp_zero, ref_zero, torch.zeros(1))
    assert torch.isclose(kl_zero, torch.tensor([[0.0]]), atol=1e-7)

    # Hand-picked x values where x = logp - ref_logp
    x_values = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    for x in x_values:
        ref_logp = torch.tensor([[0.0]])
        logp = torch.tensor([[x]])
        _, kl = compute_per_token_losses(logp, ref_logp, torch.zeros(1))
        expected_kl = math.exp(-x) + x - 1.0
        assert torch.isclose(kl, torch.tensor([[expected_kl]], dtype=torch.float32), atol=1e-6)


def test_kl_gradient_at_x_plus_two():
    """
    (c) The KL gradient at x = +2 equals 1 - e^-2 (not e^2 - 1).
    """
    ref_logp = torch.tensor([[0.0]])
    logp = torch.tensor([[2.0]], requires_grad=True)
    advantages = torch.zeros(1)

    _, kl = compute_per_token_losses(logp, ref_logp, advantages)
    kl.sum().backward()

    assert logp.grad is not None
    expected_grad = 1.0 - math.exp(-2.0)
    wrong_grad = math.exp(2.0) - 1.0
    assert torch.isclose(logp.grad, torch.tensor([[expected_grad]], dtype=torch.float32), atol=1e-5)
    assert not torch.isclose(logp.grad, torch.tensor([[wrong_grad]], dtype=torch.float32), atol=1e-2)


def test_negative_advantage_reverses_gradient_sign():
    """
    (d) Negative advantage gives a gradient of the opposite sign (+1 per token instead of -1).
    """
    ref_logp = torch.tensor([[0.0, 0.0]])
    logp = torch.zeros_like(ref_logp, requires_grad=True)
    advantages = torch.tensor([-1.0])

    pg_loss, _ = compute_per_token_losses(logp, ref_logp, advantages, clip_eps=0.2)
    pg_loss.sum().backward()

    assert logp.grad is not None
    expected_grad = torch.full_like(logp, 1.0)
    assert torch.allclose(logp.grad, expected_grad)
