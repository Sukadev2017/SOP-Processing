from __future__ import annotations


ALLOWED_STATES = {

    "SOURCE_SUPPORTED",

    "REFERENCE_DERIVED",

    "CONFLICT",

    "SME_REQUIRED",
}


class Validator:

    def validate(
        self,
        mappings,
        target_sections,
    ):

        issues = []

        expected = {

            (
                section.number
                or section.section_id
            )

            for section
            in target_sections
        }

        mapped = {

            mapping.target_id

            for mapping
            in mappings
        }

        # ---------------------------------------------
        # Every dynamically detected target section
        # must have been processed.
        # ---------------------------------------------

        for target_id in sorted(
            expected - mapped
        ):

            issues.append(
                {
                    "severity":
                        "ERROR",

                    "rule":
                        "TARGET_NOT_PROCESSED",

                    "section":
                        target_id,

                    "requires_review":
                        True,
                }
            )

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

                    or

                    mapping.evidence
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

                        "requires_review":
                            True,
                    }
                )

        return {

            "passed":
                not any(
                    issue[
                        "severity"
                    ]
                    == "ERROR"

                    for issue
                    in issues
                ),

            "target_sections_detected":
                len(
                    target_sections
                ),

            "target_sections_processed":
                len(
                    mappings
                ),

            "human_review_required":
                True,

            "issues":
                issues,
        }
