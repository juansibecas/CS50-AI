import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    p = 1
    for name in people:
        person = people[name]
        gene_n = get_gene_n_from_groups(name, one_gene, two_genes)
        if person['father'] is None and person['mother'] is None:
            p *= PROBS["gene"][gene_n]
        else:
            father = person['father']
            mother = person['mother']
            father_n = get_gene_n_from_groups(father, one_gene, two_genes)
            mother_n = get_gene_n_from_groups(mother, one_gene, two_genes)
            p_father = get_p_of_getting_gene_from_parent(father_n)
            p_mother = get_p_of_getting_gene_from_parent(mother_n)

            if gene_n == 2:
                p *= p_father*p_mother          # getting 1 from father * getting 1 from mother
            elif gene_n == 1:
                p *= (p_father*(1-p_mother) + (1-p_father)*p_mother)    # getting it from father * not getting it from mother
            else:                                                       # OR not getting it from father * getting it from mother
                p *= (1-p_father)*(1-p_mother)  # not getting it from father * not getting it from mother

        if name in have_trait:
            p *= PROBS['trait'][gene_n][True]
        else:
            p *= PROBS['trait'][gene_n][False]
    return p


def get_p_of_getting_gene_from_parent(parent_n):
    if parent_n == 2:
        return 1 - PROBS["mutation"]        # 100% chance of getting the gene * 99% chance of it not mutating
    elif parent_n == 1:
        return 0.5*(1-PROBS["mutation"]) + 0.5*(PROBS["mutation"])    # 50% chance of getting the gene * 99% chance of it not mutating
    else:                                                             # OR 50% chance of not getting the gene * 1% chance of it mutating
        return PROBS["mutation"]            # 100% chance of not getting the gene * 1% chance of it mutating


def get_gene_n_from_groups(name, one_gene, two_genes):
    if name in two_genes:
        return 2
    elif name in one_gene:
        return 1
    else:
        return 0


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for name in probabilities:
        person = probabilities[name]
        gene_n = get_gene_n_from_groups(name, one_gene, two_genes)
        trait = True if name in have_trait else False
        person["gene"][gene_n] += p
        person["trait"][trait] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for name in probabilities:
        person = probabilities[name]
        for var in person:
            dist = person[var]      # for each distribution, get the norm(sum of all elements)
            norm = 0                # and divide every element by it
            for p in dist:
                norm += dist[p]
            for p in dist:
                dist[p] /= norm


if __name__ == "__main__":
    main()
