"""
Build a high-level final review report from detailed skill reports.

The detailed report_*.md files remain available as artifacts. This script keeps
final_review_report.md concise: summary, release readiness, and conclusion only.
"""

import os
import re
from dataclasses import dataclass


@dataclass
class ReviewSection:
    skill: str
    title: str
    path: str
    content: str
    high: int
    medium: int
    low: int
    passed: bool
    missing: bool


REPORTS = [
    ("coding-standards", "Coding Standards & Conventions", "report_coding-standards.md"),
    ("security", "Security & Vulnerability", "report_security.md"),
    ("clean-code", "Code Smells & Clean Code", "report_clean-code.md"),
    ("performance", "Performance & Resource Optimization", "report_performance.md"),
    ("testability", "Testability & Reliability", "report_testability.md"),
    ("architecture", "Architecture & Business Logic", "report_architecture.md"),
]

THEME_RULES = {
    "coding-standards": [
        ("naming", "naming and convention consistency"),
        ("magic", "magic values and constant usage"),
        ("import", "import organization"),
        ("type", "type annotations"),
        ("format", "formatting and style consistency"),
    ],
    "security": [
        ("secret", "secret handling"),
        ("token", "token handling"),
        ("auth", "authentication and authorization"),
        ("input", "input validation"),
        ("injection", "injection risk"),
        ("permission", "permission boundaries"),
    ],
    "clean-code": [
        ("duplicate", "duplication"),
        ("complex", "complexity"),
        ("function", "function structure"),
        ("readability", "readability"),
        ("dead", "dead or unused code"),
    ],
    "performance": [
        ("query", "query efficiency"),
        ("loop", "loop efficiency"),
        ("cache", "caching"),
        ("memory", "memory usage"),
        ("latency", "latency risk"),
    ],
    "testability": [
        ("test", "test coverage"),
        ("mock", "mockability"),
        ("fixture", "fixtures"),
        ("edge", "edge case coverage"),
        ("reliability", "reliability"),
    ],
    "architecture": [
        ("coupling", "coupling"),
        ("dependency", "dependency boundaries"),
        ("layer", "layering"),
        ("business", "business logic placement"),
        ("interface", "interface design"),
    ],
}


def read_report(skill: str, title: str, path: str) -> ReviewSection:
    if not os.path.exists(path):
        return ReviewSection(skill, title, path, "", 0, 0, 0, False, True)

    with open(path, "r") as f:
        content = f.read().strip()

    high = count_severity(content, "HIGH")
    medium = count_severity(content, "MEDIUM")
    low = count_severity(content, "LOW")
    passed = is_pass_report(content, high + medium + low)

    return ReviewSection(skill, title, path, content, high, medium, low, passed, False)


def count_severity(content: str, severity: str) -> int:
    patterns = [
        rf"\[\s*{severity}\s*\]",
        rf"\|\s*{severity}\s*\|",
        rf"severity:\s*{severity}\b",
        rf"\*\*severity:\*\*\s*{severity}\b",
    ]
    matches = set()
    for pattern in patterns:
        for match in re.finditer(pattern, content, flags=re.IGNORECASE):
            matches.add(match.start())
    return len(matches)


def is_pass_report(content: str, severity_count: int) -> bool:
    normalized = content.lower()
    if severity_count > 0:
        return False
    pass_markers = ["pass", "no issues", "no violations", "no findings"]
    return any(marker in normalized for marker in pass_markers)


def summarize_themes(section: ReviewSection) -> str:
    if section.missing:
        return "Review report was not generated."
    if section.passed:
        return "No material issues were reported."

    normalized = section.content.lower()
    themes = [
        label
        for keyword, label in THEME_RULES.get(section.skill, [])
        if keyword in normalized
    ]

    if not themes and section.high + section.medium + section.low > 0:
        themes = ["review findings require follow-up"]
    elif not themes:
        themes = ["manual review of the detailed report is recommended"]

    return ", ".join(themes[:4]) + "."


def section_status(section: ReviewSection) -> str:
    if section.missing:
        return "Incomplete"
    if section.high > 0:
        return "Blocker"
    if section.medium > 0:
        return "Needs attention"
    if section.low > 0:
        return "Minor issues"
    if section.passed:
        return "Pass"
    return "Review required"


def release_assessment(sections: list[ReviewSection]) -> tuple[str, str]:
    missing = [section for section in sections if section.missing]
    total_high = sum(section.high for section in sections)
    total_medium = sum(section.medium for section in sections)
    security_high = next(
        (section.high for section in sections if section.skill == "security"),
        0,
    )

    if missing:
        return (
            "Not ready for production",
            "One or more required review reports are missing, so the release evidence is incomplete.",
        )
    if security_high > 0:
        return (
            "Not ready for production",
            "High-severity security findings must be resolved before release.",
        )
    if total_high > 0:
        return (
            "Not ready for production",
            "High-severity findings remain open across the review areas.",
        )
    if total_medium > 0:
        return (
            "Conditionally ready",
            "No high-severity blockers were found, but medium-severity findings should be resolved or explicitly accepted before production release.",
        )
    return (
        "Ready for production",
        "The available review reports do not show high or medium severity release blockers.",
    )


def build_summary(sections: list[ReviewSection]) -> str:
    total_high = sum(section.high for section in sections)
    total_medium = sum(section.medium for section in sections)
    total_low = sum(section.low for section in sections)
    decision, decision_reason = release_assessment(sections)

    lines = [
        "# AI Code Review Summary",
        "",
        "This is a high-level summary generated from the detailed skill reports. Detailed fixes and code-level recommendations remain in the individual `report_*.md` artifacts.",
        "",
        "## Overall Findings",
        "",
        f"- High severity findings: {total_high}",
        f"- Medium severity findings: {total_medium}",
        f"- Low severity findings: {total_low}",
        f"- Production release assessment: **{decision}**",
        "",
        "## Review Area Summary",
        "",
        "| Area | Status | Severity Summary | General Notes |",
        "|------|--------|------------------|---------------|",
    ]

    for section in sections:
        severity_summary = (
            f"H:{section.high} / M:{section.medium} / L:{section.low}"
            if not section.missing
            else "Report missing"
        )
        lines.append(
            f"| {section.title} | {section_status(section)} | {severity_summary} | {summarize_themes(section)} |"
        )

    lines.extend(
        [
            "",
            "## Production Release Assessment",
            "",
            f"**Decision:** {decision}",
            "",
            decision_reason,
            "",
            "## Conclusion",
            "",
            conclusion(decision),
            "",
        ]
    )

    return "\n".join(lines)


def conclusion(decision: str) -> str:
    if decision == "Ready for production":
        return "The reviewed changes meet the current automated review threshold for production release. Continue with normal release validation, including CI, deployment checks, and any required manual approvals."
    if decision == "Conditionally ready":
        return "The release can proceed only if the remaining medium-severity items are resolved or explicitly accepted by the owning team. Do not treat this as an unconditional production approval."
    return "The release should not proceed to production until the blocking review gaps or high-severity findings are addressed and the review is rerun successfully."


def main() -> None:
    sections = [read_report(*report) for report in REPORTS]
    summary = build_summary(sections)

    with open("final_review_report.md", "w") as f:
        f.write(summary)

    print(summary)


if __name__ == "__main__":
    main()
