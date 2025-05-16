import os
from glob import glob
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.query import Term, Phrase, Prefix, Wildcard, And, FuzzyTerm, Or, Not

folder_path = "sample_data"
file_paths = glob(os.path.join(folder_path, "*.txt"))

index_dir = "query_index"

def create_schema():
    schema = Schema(
        title=ID(stored=True),
        path=ID(stored=True),
        content=TEXT(stored=True),
    )

    return schema


def create_or_open_index(schema):
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)
        index = create_in(index_dir, schema)
    else:
        index = open_dir(index_dir)
    return index



def add_documents_to_index(index, file_paths):
    writer = index.writer()
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            title = os.path.basename(file_path)
            writer.add_document(title=title, path=file_path, content=content)
    writer.commit()
    print(f"Indexed {len(file_paths)} documents.")


def search_index(index, query):
    with index.searcher() as searcher:
        results = searcher.search(query)
        print(f"Found {len(results)} results for query: {query}")
        for hit in results:
            print(f"Found: {hit['title']} at {hit['path']}")
            print(f"Score: {hit.score}")
            print(f"Path: {hit['path']}")
            print(f"Content: {hit['content']}")
            print("------")

def term_search(index, term):
    query = Term("content", term)
    search_index(index, query)


def phrase_search(index, phrase):
    query = Phrase("content", phrase)
    search_index(index, query)


def prefix_search(index, prefix):
    query = Prefix("content", prefix)
    search_index(index, query)


def wildcard_search(index, wildcard):
    query = Wildcard("content", wildcard)
    search_index(index, query)


def fuzzy_search(index, term):
    query = FuzzyTerm("content", term)
    search_index(index, query)

def and_search(index, terms):
    all_terms = []
    for term in terms:
        all_terms.append(Term("content", term))

    query = And(all_terms)
    search_index(index, query)


def or_search(index, terms):
    all_terms = []
    for term in terms:
        all_terms.append(Term("content", term))

    query = Or(all_terms)
    search_index(index, query)


def not_search(index, terms):
    all_terms = []
    for term in terms:
        all_terms.append(Term("content", term))

    query = Not(And(all_terms))
    search_index(index, query)


def clear_index():
    if os.path.exists(index_dir):
        for file in os.listdir(index_dir):
            file_path = os.path.join(index_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(index_dir)
        print("Index cleared.")


def main():
    schema = create_schema()
    index = create_or_open_index(schema)
    add_documents_to_index(index, file_paths)
    print("Indexing completed.")
    print("Performing searches...")

    print("Select a type of query:")
    print("1. Term Search")
    print("2. Phrase Search")
    print("3. Prefix Search")
    print("4. Wildcard Search")
    print("5. Fuzzy Search")
    print("6. And Search")
    print("7. Or Search")
    print("8. Not Search")

    query_type = input("Enter the number of the query type (1-8): ")
    query_input = input("Enter the search term: ")

    if query_type == "1":
        term_search(index, query_input)
    elif query_type == "2":
        phrase_search(index, query_input)
    elif query_type == "3":
        prefix_search(index, query_input)
    elif query_type == "4":
        wildcard_search(index, query_input)
    elif query_type == "5":
        fuzzy_search(index, query_input)
    elif query_type == "6":
        terms = query_input.split()
        and_search(index, terms)
    elif query_type == "7":
        terms = query_input.split()
        or_search(index, terms)
    elif query_type == "8":
        terms = query_input.split()
        not_search(index, terms)
    else:
        print("Invalid query type selected.")

    print("Search completed.")

if __name__ == "__main__":
    clear_index()
    main()