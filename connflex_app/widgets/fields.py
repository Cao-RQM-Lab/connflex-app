"""Input fields for physical quantities typed with SI prefixes."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLineEdit, QToolTip, QWidget

from connflex_app.units import fmt_si, parse_si


class QuantityEdit(QLineEdit):
    """A line edit holding one quantity, typed as ``3 µm``, ``3e-6``, or ``3u``.

    With ``optional`` the field may be left empty, which means "choose for me"; the
    placeholder then shows what would be chosen. A value that cannot be read, or that lies
    outside the allowed range, is refused with a tooltip and the last good value comes back.
    """

    valueChanged = Signal()

    def __init__(
        self,
        unit: str = "",
        value: float | None = None,
        *,
        optional: bool = False,
        minimum: float | None = None,
        maximum: float | None = None,
        positive: bool = False,
        digits: int = 4,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._unit = unit
        self._optional = optional
        self._minimum = minimum
        self._maximum = maximum
        self._positive = positive
        self._digits = digits
        self._value: float | None = value
        self._automatic = ""
        self.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.setMinimumWidth(96)
        self.editingFinished.connect(self._commit)
        self._show()

    def value(self) -> float | None:
        return self._value

    def set_value(self, value: float | None) -> None:
        self._value = value
        self._show()

    def set_automatic(self, text: str) -> None:
        """What an empty field stands for, shown as its placeholder, e.g. ``auto · 1.5 µm``."""
        self._automatic = text
        self._show()

    def _format(self, value: float) -> str:
        if not self._unit:  # a bare number reads better as 1e-06 than as 1 µ
            return f"{value:.{self._digits}g}"
        return fmt_si(value, self._unit, self._digits)

    def _show(self) -> None:
        self.blockSignals(True)
        self.setText("" if self._value is None else self._format(self._value))
        self.setPlaceholderText(self._automatic if self._optional else "")
        self.blockSignals(False)

    def _refuse(self, message: str) -> None:
        QToolTip.showText(self.mapToGlobal(self.rect().bottomLeft()), message, self)
        self._show()

    def _commit(self) -> None:
        text = self.text().strip()
        if not text or text.lower().startswith("auto"):
            if self._optional:
                changed = self._value is not None
                self._value = None
                self._show()
                if changed:
                    self.valueChanged.emit()
            else:
                self._show()
            return
        try:
            value = parse_si(text, self._unit)
        except ValueError as exc:
            self._refuse(str(exc))
            return
        if self._positive and value <= 0:
            self._refuse("enter a value above zero")
            return
        if self._minimum is not None and value < self._minimum:
            self._refuse(f"enter at least {self._format(self._minimum)}")
            return
        if self._maximum is not None and value > self._maximum:
            self._refuse(f"enter at most {self._format(self._maximum)}")
            return
        changed = value != self._value
        self._value = value
        self._show()
        if changed:
            self.valueChanged.emit()
