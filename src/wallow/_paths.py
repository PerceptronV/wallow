"""Path-component sanitisation and layout substitution.

Used by ``Store.artefacts_dir`` to turn a free-form layout template like
``"{experiment_type}/{uuid}"`` into a filesystem-safe relative path. Pure
stdlib, no external dependencies.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping

from .errors import SchemaValidationError

# Single open-curly token — same one Schema.validate_layout uses.
_LAYOUT_PLACEHOLDER = re.compile(r"\{([^{}]+)\}")
# Drop anything outside this whitelist after ASCII normalisation.
_PATH_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitise_for_path(value: Any) -> str:
    """Convert *value* into a filesystem-safe path component.

    Steps:
      1. ``str(value)``.
      2. NFKD-normalise and drop non-ASCII bytes (so ``café`` → ``cafe``).
      3. Collapse runs of whitespace + illegal chars into a single underscore
         (anything outside ``[A-Za-z0-9._-]``).
      4. Strip leading/trailing dots and underscores so the component is
         neither hidden (``.foo``) nor empty.

    Raises ``ValueError`` if the result is empty after sanitisation — the
    caller should usually wrap this in a clearer error mentioning the field.
    """
    text = str(value)
    # NFKD splits accented chars into base + combining marks; ASCII drop
    # then keeps only the base char.
    normalised = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    cleaned = _PATH_SAFE.sub("_", normalised).strip("._")
    if not cleaned:
        raise ValueError(
            f"sanitise_for_path({value!r}) produced an empty component"
        )
    return cleaned


def substitute_layout(layout: str, attrs: Mapping[str, Any]) -> str:
    """Substitute ``{name}`` placeholders in *layout* with sanitised attrs.

    *attrs* is a mapping of field name → value (typically pulled off a Run
    via ``getattr``). Each substituted value is run through
    ``sanitise_for_path``. Unknown placeholders raise
    ``SchemaValidationError`` — though the schema-load-time
    ``Schema.validate_layout`` call usually catches these earlier.
    """
    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in attrs:
            raise SchemaValidationError(
                f"layout placeholder {{{name}}} has no value on this run; "
                f"available: {sorted(attrs)}",
                field=name,
            )
        try:
            return sanitise_for_path(attrs[name])
        except ValueError as e:
            raise SchemaValidationError(
                f"layout placeholder {{{name}}} sanitised to empty: {e}",
                field=name,
            ) from None

    return _LAYOUT_PLACEHOLDER.sub(_replace, layout)
