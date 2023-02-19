import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    all_pages = corpus.keys()
    reachable_pages = corpus[page]

    model = {}

    for page2 in all_pages:
        p = (1-damping_factor)/len(all_pages)       # p of picking a random page, given page has links
        if len(reachable_pages) == 0:               # p of picking a random page, no links
            p = 1/len(all_pages)
        elif page2 in reachable_pages:
            p += damping_factor/len(reachable_pages) # p of going to page2 from a link in page
        model[page2] = p

    return model


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    all_pages = list(corpus.keys())
    sample = [random.choice(all_pages)]
    for i in range(n):
        model = transition_model(corpus, sample[i], damping_factor)
        weighted_population = random.choices(list(model.keys()), weights=model.values(), k=1000)
        sample.append(random.choice(weighted_population))

    pagerank = {}
    for i in range(n):  # adds 1 for each time page appeared in sample
        try:
            pagerank[sample[i]] += 1
        except:
            pagerank[sample[i]] = 1

    for page in all_pages:
        pagerank[page] = pagerank[page]/n

    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    tol = 0.001
    pagerank = {}
    n_pages = len(corpus.keys())
    for page in corpus.keys():
        pagerank[page] = 1/n_pages
    it = 0
    while True:
        it += 1
        tol_counter = 0
        for page1 in corpus.keys():
            sigma = 0       # sum in pagerank formula
            for page2 in corpus.keys():
                if page1 == page2:
                    continue
                else:
                    if len(corpus[page2]) == 0:         # if no links on page2
                        sigma += pagerank[page2]/n_pages
                    elif page1 in corpus[page2]:        # if theres a link to page1 on page2
                        sigma += pagerank[page2]/len(corpus[page2])
                    else:
                        sigma += 0

            new_pagerank = (1-damping_factor)/n_pages + damping_factor*sigma
            if abs(pagerank[page1] - new_pagerank) < tol:
                tol_counter += 1        # plus 1 for each pagerank value change below tol
            pagerank[page1] = new_pagerank
        if tol_counter == n_pages:
            break

    return pagerank


if __name__ == "__main__":
    main()