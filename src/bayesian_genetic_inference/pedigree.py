"""Loading and checking family pedigrees.

CSV format (same as CS50 AI Heredity): columns name, mother, father, trait
* mother/father are blank for founders (people with no recorded parents);
* trait is 1 (has the trait), 0 (does not) or blank (unknown).

A pedigree is rejected if it has duplicate names, a parent who is not in the
file, exactly one recorded parent, a person who is their own parent, or a
circular relationship (someone being their own ancestor).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


class PedigreeError(ValueError):
    """Raised for any structurally invalid pedigree."""


@dataclass(frozen=True)
class Person:
    name: str
    mother: str | None
    father: str | None
    observed_trait: bool | None     # None = trait not observed

    @property
    def is_founder(self) -> bool:
        return self.mother is None and self.father is None


@dataclass(frozen=True)
class Pedigree:
    """A validated family. `people` is in topological order (parents before children)."""

    people: tuple[Person, ...]

    @property
    def names(self) -> list[str]:
        return [person.name for person in self.people]

    def parent_map(self) -> dict[str, tuple[str, str] | None]:
        return {p.name: None if p.is_founder else (p.mother, p.father) for p in self.people}

    @property
    def observed_traits(self) -> dict[str, bool]:
        return {p.name: p.observed_trait for p in self.people if p.observed_trait is not None}

    @property
    def unobserved_trait_names(self) -> list[str]:
        return [p.name for p in self.people if p.observed_trait is None]


def build_pedigree(people: list[Person]) -> Pedigree:
    """Validate a list of people and return them as a Pedigree in topological order."""
    if not people:
        raise PedigreeError("pedigree is empty")
    by_name: dict[str, Person] = {}
    for person in people:
        if not person.name:
            raise PedigreeError("every person needs a non-empty name")
        if person.name in by_name:
            raise PedigreeError(f"duplicate person {person.name!r}")
        by_name[person.name] = person

    for person in people:
        if (person.mother is None) != (person.father is None):
            raise PedigreeError(f"{person.name!r} has only one recorded parent; record both or neither")
        for parent in (person.mother, person.father):
            if parent is None:
                continue
            if parent == person.name:
                raise PedigreeError(f"{person.name!r} is listed as their own parent")
            if parent not in by_name:
                raise PedigreeError(f"parent {parent!r} of {person.name!r} is not in the pedigree")
        if person.mother is not None and person.mother == person.father:
            raise PedigreeError(f"{person.name!r} has the same person as mother and father")

    return Pedigree(tuple(_topological_order(by_name)))


def _topological_order(by_name: dict[str, Person]) -> list[Person]:
    """Depth-first search that puts parents before children and detects cycles."""
    ordered: list[Person] = []
    state: dict[str, str] = {}          # "visiting" while on the DFS stack, "done" afterwards

    def visit(name: str, path: list[str]) -> None:
        if state.get(name) == "done":
            return
        if state.get(name) == "visiting":
            cycle = " -> ".join(path[path.index(name):] + [name])
            raise PedigreeError(f"circular relationship: {cycle}")
        state[name] = "visiting"
        person = by_name[name]
        for parent in (person.mother, person.father):
            if parent is not None:
                visit(parent, path + [name])
        state[name] = "done"
        ordered.append(person)

    for name in by_name:                 # keep the file order where possible
        visit(name, [])
    return ordered


def _parse_trait(raw: str, name: str) -> bool | None:
    value = raw.strip()
    if value == "":
        return None
    if value == "1":
        return True
    if value == "0":
        return False
    raise PedigreeError(f"trait for {name!r} must be 1, 0 or blank, got {raw!r}")


def load_pedigree_csv(path: str | Path) -> Pedigree:
    """Read a CS50-style pedigree CSV and validate it."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {"name", "mother", "father", "trait"}
        if reader.fieldnames is None or set(reader.fieldnames) != expected:
            raise PedigreeError(f"CSV columns must be exactly {sorted(expected)}, got {reader.fieldnames}")
        people = []
        for row in reader:
            name = row["name"].strip()
            people.append(Person(
                name=name,
                mother=row["mother"].strip() or None,
                father=row["father"].strip() or None,
                observed_trait=_parse_trait(row["trait"], name),
            ))
    return build_pedigree(people)
