import json
from src.rl_env.observation import Observation, ObservationSerializer
from src.rl_env.env import Intent
from src.repo_scan.models import RepoManifest, EcosystemManifest

def test_to_chat_prompt_structure_and_content():
    """
    Verify that `to_chat_prompt` returns a list of message dicts with
    'system' and 'user' roles, and that the content accurately reflects
    the vocabulary in the system prompt and the ecosystem/conflict details
    in the user prompt.
    """
    manifest = RepoManifest(
        ecosystems={
            "python": EcosystemManifest(
                ecosystem="python",
                manifest_files=["requirements.txt"],
                dependencies={"requests": None},
            )
        },
        conflicts=["Detected version conflict for requests"]
    )

    obs = Observation(
        canonical_intent=Intent(
            intent_text="Install requirements",
            template_name="setup_project",
            repo_id="owner/repo"
        ),
        repo_manifest=manifest,
        step_index=0,
        max_steps=10,
        episode_steps=[]
    )

    serializer = ObservationSerializer()
    prompt = serializer.to_chat_prompt(obs)

    assert isinstance(prompt, list)
    assert len(prompt) == 2
    assert prompt[0]["role"] == "system"
    assert prompt[1]["role"] == "user"

    system_content = prompt[0]["content"]
    user_content = prompt[1]["content"]

    # Verify action vocabulary is in system prompt
    assert "command" in system_content
    assert "description" in system_content
    assert "verify_command" in system_content
    assert "destructive" in system_content

    # Verify user prompt includes key information
    assert "Install requirements" in user_content
    assert "setup_project" in user_content
    assert "owner/repo" in user_content
    assert "python:" in user_content
    assert "manifest present" in user_content
    assert "Detected version conflict for requests" in user_content
    assert "Step 1 of 10" in user_content
