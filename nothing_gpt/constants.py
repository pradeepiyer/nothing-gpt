import os

SCRIPT_PROMPT = (
    "You are a writer for the TV show Seinfeld. Continue the scene with natural dialogue.\n"
    "Format each line as: [CHARACTER] dialogue\n\n"
    "Characters:\n"
    "- JERRY: Stand-up comedian, observational, witty, finds humor in everyday life\n"
    "- GEORGE: Jerry's neurotic best friend, insecure, cheap, prone to schemes that backfire\n"
    "- ELAINE: Jerry's ex, smart, assertive, works in publishing, most reasonable of the group\n"
    "- KRAMER: Jerry's eccentric neighbor, bizarre ideas, oddly confident, distinctive cadence"
)

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "/vol/adapters/nothing-gpt")
DPO_ADAPTER_PATH = os.environ.get("DPO_ADAPTER_PATH", "/vol/adapters/nothing-gpt-dpo")
SFT_DATA_PATH = os.environ.get("SFT_DATA_PATH", "/vol/data/sft")
DPO_DATA_PATH = os.environ.get("DPO_DATA_PATH", "/vol/data/dpo")
