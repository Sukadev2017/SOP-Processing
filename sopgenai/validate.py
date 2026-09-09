from __future__ import annotations

from .models import (
    Mapping,
)


REQUIRED_SECTIONS = {
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


ALLOWED_STATES = {
    "SOURCE_SUPPORTED",
    "REFERENCE_DERIVED",
    "CONFLICT",
    "SME_REQUIRED",
}


class Validator:

    def validate(
        self,
        mappings: list[
            Mapping
        ],
    ):

        issues = []

        populated = {
            mapping.target_id
            for mapping
            in mappings
            if (
                mapping
                .transformed_content
                .strip()
            )
        }

        # --------------------------------
        # Required sections
        # --------------------------------

        for section_id in sorted(
            REQUIRED_SECTIONS
            - populated
        ):

            issues.append(
                {
                    "severity":
                        "WARNING",

                    "rule":
                        "MISSING_SECTION",

                    "section":
                        section_id,

                    "message":
                        "Required template "
                        "section has no "
                        "generated content.",

                    "requires_review":
                        True,
                }
            )

        # --------------------------------
        # Mapping validation
        # --------------------------------

        for mapping in mappings:

            if (
                mapping.state
                not in ALLOWED_STATES
            ):

                issues.append(
                    {
                        "severity":
                            "ERROR",

                        "rule":
                            "INVALID_STATE",

                        "section":
                            mapping.target_id,

                        "message":
                            "Invalid generation "
                            "state.",

                        "requires_review":
                            True,
                    }
                )

            if mapping.state in {
                "CONFLICT",
                "SME_REQUIRED",
            }:

                issues.append(
                    {
                        "severity":
                            "WARNING",

                        "rule":
                            mapping.state,

                        "section":
                            mapping.target_id,

                        "message":
                            "Human review "
                            "is required.",

                        "requires_review":
                            True,
                    }
                )

            if (
                mapping.state
                in {
                    "SOURCE_SUPPORTED",
                    "REFERENCE_DERIVED",
                }
                and not (
                    mapping
                    .source_element_ids
                    or mapping.evidence
                )
            ):

                issues.append(
                    {
                        "severity":
                            "ERROR",

                        "rule":
                            "MISSING_PROVENANCE",

                        "section":
                            mapping.target_id,

                        "message":
                            "Generated content "
                            "does not contain "
                            "source provenance.",

                        "requires_review":
                            True,
                    }
                )

        passed = not any(
            issue[
                "severity"
            ]
            == "ERROR"
            for issue
            in issues
        )

        return {
            "passed":
                passed,

            "human_review_required":
                True,

            "issues":
                issues,
        }
