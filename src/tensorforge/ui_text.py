"""English-only interface copy. Dataset language is configured independently."""

ENGLISH = {
    "badge": "LOCAL / OFFLINE / HUMAN REVIEWED",
    "intro": "Turn competition data into an editable baseline notebook.",
    "independent": "Independent project. No affiliation with Google's TensorFlow. Generation never "
    "trains or runs models.",
    "verified": "Verified core adapters",
    "experimental": "Experimental templates",
    "catalog": "Scenario catalog",
    "tabs": ["Build a notebook", "Inspect data", "Task coverage", "Environment & languages"],
    "workflow": "Review → validate → generate",
    "start": "Start with an example YAML. Data paths resolve from its project_root.",
    "config": "Configuration file",
    "output": "New bundle directory",
    "allow": "Allow experimental local-model templates",
    "validate": "Validate configuration",
    "generate": "Generate notebook",
    "valid": "Schema and local prerequisites validated. Review warnings before approving the YAML.",
    "generated": "Notebook generated. No training or inference was performed.",
    "download": "Download notebook",
    "execution": "Execution is explicit: open the notebook from the bundle folder or run verify in "
    "your terminal.",
    "manifest": "Local CSV, JSONL or Parquet manifest",
    "limit": "Maximum inspected rows",
    "inspect": "Inspect manifest",
    "status": "Implementation status",
    "coverage_note": "Planned tasks fail explicitly. Experimental templates require compatible local "
    "assets and are not declared model-verified.",
    "setup": "Setup/offline guide: README.md and docs/PORTABILITY.md. Dataset languages: docs/LANGUAGES.md.",
    "languages": "Indonesian, English and regional languages",
    "language_note": "Character baselines accept Unicode and code-switching. OCR, fonts and "
    "pretrained-model language/script coverage must be checked independently.",
}
