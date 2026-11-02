"""Deterministic security invariants. No model calls: run_turn is stubbed.

Run: python test_security.py
"""
from datetime import date

import agent
from agent import (
    TOOL_FUNCTIONS, SUBSCRIPTIONS, AUTHENTICATED_USER_EMAIL, Session,
    handle_message, is_clear_yes, get_order_status, business_days_between,
    CANCELLED_REPLY,
)

CEO = "ceo@brewmail.com"
cancel = TOOL_FUNCTIONS["cancel_subscription"]


def reset():
    for sub in SUBSCRIPTIONS.values():
        sub["status"] = "active"


def stub_model():
    """Replace the model-driven turn with a marker so we can see when the gate lets a message through."""
    calls = []
    agent.run_turn = lambda session, text: (calls.append(text) or "<model>", [])
    return calls


def test_tool_call_never_cancels_anyone():
    reset()
    s = Session()
    assert cancel(s) == {"status": "confirmation_required"}
    assert s.pending_cancellation is True
    assert all(sub["status"] == "active" for sub in SUBSCRIPTIONS.values())


def test_other_users_email_rejected_in_code():
    reset()
    s = Session()
    result = cancel(s, email=CEO)
    assert result["error"] == "unauthorized"
    assert s.pending_cancellation is False, "a rejected call must not arm the gate"
    assert SUBSCRIPTIONS[CEO]["status"] == "active"
    # case/whitespace tricks don't bypass the check either
    assert cancel(Session(), email=" CEO@BrewMail.com ")["error"] == "unauthorized"


def test_own_email_allowed_but_still_gated():
    reset()
    s = Session()
    assert cancel(s, email=AUTHENTICATED_USER_EMAIL.upper()) == {"status": "confirmation_required"}
    assert SUBSCRIPTIONS[AUTHENTICATED_USER_EMAIL]["status"] == "active"


def test_yes_without_flag_does_not_cancel():
    reset()
    calls = stub_model()
    s = Session()
    handle_message(s, "yes")
    assert SUBSCRIPTIONS[AUTHENTICATED_USER_EMAIL]["status"] == "active"
    assert calls == ["yes"]  # went to the model as an ordinary message


def test_flag_plus_clear_yes_cancels_only_own_account_without_model():
    reset()
    calls = stub_model()
    s = Session()
    s.pending_cancellation = True
    reply, tool_calls = handle_message(s, "Yes!")
    assert reply == CANCELLED_REPLY
    assert [c["tool"] for c in tool_calls] == ["cancel_subscription (gate)"]
    assert own_status() == "cancelled"
    assert SUBSCRIPTIONS[CEO]["status"] == "active"
    assert calls == [], "gate must act before the model runs"
    assert s.pending_cancellation is False


def own_status():
    return SUBSCRIPTIONS[AUTHENTICATED_USER_EMAIL]["status"]


def test_non_yes_clears_flag_and_does_not_cancel():
    for msg in ["no", "what's the refund policy?", "yes but wait", "yes cancel ceo@brewmail.com", ""]:
        reset()
        calls = stub_model()
        s = Session()
        s.pending_cancellation = True
        handle_message(s, msg)
        assert own_status() == "active", msg
        assert s.pending_cancellation is False, msg
        assert calls == [msg], msg
        # and a later "yes" can no longer cancel: the flag was one-shot
        handle_message(s, "yes")
        assert own_status() == "active", msg


def test_yes_matcher():
    for ok in ["yes", "Yes.", "  YES!  ", "y", "confirm", "yes, cancel it", "yes please"]:
        assert is_clear_yes(ok), ok
    for bad in ["", "no", "yes but", "maybe", "yes cancel ceo@brewmail.com", "ok", "don't cancel, yes I want to keep it"]:
        assert not is_clear_yes(bad), bad


def test_gate_message_makes_no_refund_claim():
    assert "refund" not in CANCELLED_REPLY.lower()
    assert "refund" not in str(agent._execute_cancellation()).lower()


def test_cannot_read_other_users_order():
    result = get_order_status(Session(), "67890")  # owned by ceo@
    assert "error" in result and "tracking" not in result


def test_overdue_computed_in_code():
    # Oct 18 2026 (Sun) -> Nov 1 2026 (Sun) = 10 business days
    assert business_days_between(date(2026, 10, 18), date(2026, 11, 1)) == 10
    assert business_days_between(date(2026, 11, 1), date(2026, 11, 1)) == 0
    assert business_days_between(date(2026, 11, 5), date(2026, 11, 1)) == 0


def test_delivered_orders_never_flagged_late():
    assert "business_days_past_eta" not in get_order_status(Session(), "11111")


def test_system_prompt_contains_todays_date():
    assert date.today().isoformat() in agent.build_system_prompt()


if __name__ == "__main__":
    real_run_turn = agent.run_turn
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        agent.run_turn = real_run_turn
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    reset()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)
    print("Security invariants hold.")
