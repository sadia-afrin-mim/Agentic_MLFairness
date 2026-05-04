

# ─────────────────────────────────────────────────────────────
# OPTION 2: Full control (specify everything)
# Uncomment and use this if you want full control
# ─────────────────────────────────────────────────────────────

# agent = FairAgent(
#     # Required
#     dataset_path      = "bank-full.csv",
#     target_col        = "y",
#     api_key           = "sk-...",
#
#     # LLM settings
#     llm_model         = "gpt-4o-mini",   # or "gpt-4o"
#
#     # Model settings
#     model_type        = "rf",   # "rf", "lr", or "gb"
#
#     # Protected attributes (skip to auto-detect)
#     protected_attrs   = ["age", "marital", "education"],
#
#     # Search settings
#     n_seeds           = 20,    # seeds per iteration
#     n_counterfactuals = 10,    # counterfactuals per seed
#     max_retries       = 2,     # retries on failure
#     n_iterations      = 3,     # like CoverUp's 3 runs
#
#     # Output
#     fairness_threshold= 0.1,
#     output_dir        = "fairagent_results"
# )
#
# results = agent.run()
#
# print(f"\nDone! Found {results.good} discriminatory pairs")
# print(f"Cost: ${results.total_cost:.3f}")
from fairagent import FairAgent

agent = FairAgent(
    # Required
    dataset_path      = "bank-small-biased.csv",
    target_col        = "y",
    api_key      = "sk..",          # ← your OpenAI key

    # LLM
    llm_model         = "gpt-4o-mini",

    # Model
    model_type        = "rf",

    # Protected attributes (skip prompts)
    protected_attrs   = ["age", "marital"],

    # Search settings (stronger search)
    n_seeds           = 5,    # was 20
    n_counterfactuals = 3,    # was 10
    max_retries       = 0,    # was 2
    n_iterations      = 1,    # was 3

    # 🔴 IMPORTANT: lower threshold
    fairness_threshold= 0.0,

    output_dir        = "fairagent_results"
)

results = agent.run()

print(f"\nDone! Found {results.good} discriminatory pairs")
print(f"Cost: ${results.total_cost:.3f}")