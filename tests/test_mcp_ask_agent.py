"""Test that ask_agent properly injects sender identity."""

import pytest
from claudechic.mcp import _make_ask_agent, set_app


class MockAgent:
    def __init__(self, name: str):
        self.name = name
        self.id = name
        self.cwd = "/tmp"
        self.status = "idle"
        self.worktree = None
        self.client = True  # truthy
        self.received_prompt = None

    async def send(self, prompt: str) -> None:
        self.received_prompt = prompt


class MockAgentManager:
    def __init__(self):
        self.agents: dict[str, MockAgent] = {}

    def add(self, agent: MockAgent) -> None:
        self.agents[agent.name] = agent

    def find_by_name(self, name: str) -> MockAgent | None:
        return self.agents.get(name)

    def __len__(self) -> int:
        return len(self.agents)


class MockApp:
    def __init__(self):
        self.agent_mgr = MockAgentManager()


@pytest.fixture
def mock_app():
    app = MockApp()
    set_app(app)  # type: ignore
    return app


@pytest.mark.asyncio
async def test_ask_agent_injects_sender(mock_app):
    """When agent 'alice' asks agent 'bob' a question, bob should see it's from alice."""
    alice = MockAgent("alice")
    bob = MockAgent("bob")
    mock_app.agent_mgr.add(alice)
    mock_app.agent_mgr.add(bob)

    # Create ask_agent tool bound to alice
    ask_agent = _make_ask_agent(caller_name="alice")

    # Call the handler directly
    await ask_agent.handler({"name": "bob", "prompt": "What's the weather?"})

    # Bob should have received the prompt with alice's identity and reply instruction
    assert bob.received_prompt is not None
    assert "[Question from agent 'alice'" in bob.received_prompt
    assert "please respond back using tell_agent" in bob.received_prompt
    assert "What's the weather?" in bob.received_prompt


@pytest.mark.asyncio
async def test_ask_agent_without_sender(mock_app):
    """When no sender is specified (legacy), prompt should pass through unchanged."""
    bob = MockAgent("bob")
    mock_app.agent_mgr.add(bob)

    # Create ask_agent tool without caller name
    ask_agent = _make_ask_agent()

    await ask_agent.handler({"name": "bob", "prompt": "What's the weather?"})

    # Without sender, prompt should be unchanged
    assert bob.received_prompt == "What's the weather?"
