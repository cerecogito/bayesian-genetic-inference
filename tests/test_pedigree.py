"""Tests for pedigree loading and validation."""

import pytest

from bayesian_genetic_inference.pedigree import PedigreeError, Person, build_pedigree, load_pedigree_csv


def write(tmp_path, text):
    path = tmp_path / "family.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_synthetic_pedigree_in_topological_order():
    pedigree = load_pedigree_csv("data/synthetic_pedigree_8.csv")
    assert len(pedigree.people) == 8
    assert pedigree.observed_traits == {"Alan": True, "Fiona": False, "Grace": True}
    assert len(pedigree.unobserved_trait_names) == 5
    position = {name: i for i, name in enumerate(pedigree.names)}
    for person in pedigree.people:
        if not person.is_founder:
            assert position[person.mother] < position[person.name] > position[person.father]


def test_child_listed_before_parents_is_reordered():
    pedigree = load_pedigree_csv("data/small_family_3.csv")      # Harry is the first row
    assert pedigree.names.index("Harry") == 2


def test_missing_parent_is_rejected(tmp_path):
    with pytest.raises(PedigreeError, match="not in the pedigree"):
        load_pedigree_csv(write(tmp_path, "name,mother,father,trait\nKid,Mum,Dad,\nMum,,,\n"))


def test_single_recorded_parent_is_rejected(tmp_path):
    with pytest.raises(PedigreeError, match="only one recorded parent"):
        load_pedigree_csv(write(tmp_path, "name,mother,father,trait\nKid,Mum,,\nMum,,,\n"))


def test_circular_relationship_is_rejected():
    people = [Person("A", "B", "C", None), Person("B", "A", "C", None), Person("C", None, None, None)]
    with pytest.raises(PedigreeError, match="circular"):
        build_pedigree(people)


def test_own_parent_and_duplicates_are_rejected():
    with pytest.raises(PedigreeError, match="own parent"):
        build_pedigree([Person("A", "A", "B", None), Person("B", None, None, None)])
    with pytest.raises(PedigreeError, match="duplicate"):
        build_pedigree([Person("A", None, None, None), Person("A", None, None, None)])
    with pytest.raises(PedigreeError, match="same person"):
        build_pedigree([Person("K", "P", "P", None), Person("P", None, None, None)])


def test_bad_trait_value_and_columns(tmp_path):
    with pytest.raises(PedigreeError, match="1, 0 or blank"):
        load_pedigree_csv(write(tmp_path, "name,mother,father,trait\nA,,,yes\n"))
    with pytest.raises(PedigreeError, match="columns"):
        load_pedigree_csv(write(tmp_path, "name,mum,dad,trait\nA,,,\n"))
