from hybrid_retriever import retrieve
from llm import generate_answer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TOP_K = 5

MIN_HYBRID_SCORE = 0.30


# ---------------------------------------------------------
# Check whether we have enough evidence
# ---------------------------------------------------------

def check_evidence(results):

    if not results:
        return False

    best_score = results[0]["hybrid_score"]

    if best_score < MIN_HYBRID_SCORE:
        return False

    return True


# ---------------------------------------------------------
# Format retrieved evidence
# ---------------------------------------------------------

def build_evidence(results):

    evidence = []

    for i, result in enumerate(results, start=1):

        evidence.append(
            f"""
SOURCE {i}

Document:
{result["document"]}

Policy ID:
{result["policy_id"]}

Section:
{result["section"]}

Page:
{result["pages"]}

Content:
{result["text"]}
"""
        )

    return "\n".join(evidence)


# ---------------------------------------------------------
# Main RAG pipeline
# ---------------------------------------------------------

def answer_question(question):

    print("\nSearching policy documents...")

    results = retrieve(
        question,
        top_k=TOP_K
    )


    # -----------------------------------------------------
    # Evidence check
    # -----------------------------------------------------

    if not check_evidence(results):

        return {
            "answer": (
                "I don't have enough information in "
                "the hospital policy documents to "
                "answer that question."
            ),
            "sources": [],
            "results": results
        }


    # -----------------------------------------------------
    # Build evidence
    # -----------------------------------------------------

    evidence = build_evidence(results)


    # -----------------------------------------------------
    # Generate answer using LLM
    # -----------------------------------------------------

    answer = generate_answer(
        question,
        evidence
    )


    return {
        "answer": answer,
        "sources": results,
        "results": results
    }


# ---------------------------------------------------------
# Interactive testing
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("CAREASSIST POLICY ASSISTANT")
    print("=" * 70)

    while True:

        question = input(
            "\nAsk a hospital policy question "
            "(or type 'exit'): "
        )


        if question.lower() == "exit":
            break


        response = answer_question(
            question
        )


        # -------------------------------------------------
        # Display answer
        # -------------------------------------------------

        print("\n")
        print("=" * 70)
        print("ANSWER")
        print("=" * 70)

        print(
            response["answer"]
        )


        # -------------------------------------------------
        # Display sources
        # -------------------------------------------------

        if response["sources"]:

            print("\n")
            print("=" * 70)
            print("SOURCES")
            print("=" * 70)


            for source in response["sources"][:3]:

                print(
                    f"\n{source['document']}"
                )

                print(
                    f"Section: "
                    f"{source['section']}"
                )

                print(
                    f"Page: "
                    f"{source['pages']}"
                )

                print(
                    f"Score: "
                    f"{source['hybrid_score']:.4f}"
                )