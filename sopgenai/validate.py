from __future__ import annotations


MANDATORY = {
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "10",
}


class DeterministicValidator:

    def validate(
        self,
        mapping,
    ):

        issues = []

        mappings = mapping.get(
            "mappings",
            []
        )

        available = {
            str(item.get(
                "target_section_id"
            ))
            for item in mappings
        }

        for required in MANDATORY:

            if required not in available:

                issues.append(
                    {
                        "rule":
                            "MANDATORY_SECTION",

                        "severity":
                            "ERROR",

                        "section":
                            required,

                        "message":
                            "Mandatory template "
                            "section has no mapping.",
                    }
                )

        for item in mappings:

            if (
                item.get(
                    "generation_state"
                )
                == "AI_INFERRED"
            ):

                issues.append(
                    {
                        "rule":
                            "AI_INFERENCE",

                        "severity":
                            "REVIEW",

                        "section":
                            item.get(
                                "target_section_id"
                            ),

                        "message":
                            "AI-inferred content "
                            "requires SME review.",
                    }
                )

        return {
            "passed": not any(
                x["severity"] == "ERROR"
                for x in issues
            ),

            "issues": issues,
        }
